package pull

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

func reusableStagedFile(root *os.Root, name string, f File) bool {
	info, err := root.Lstat(name)
	if err != nil || !info.Mode().IsRegular() || info.Size() != f.Size {
		return false
	}
	if f.SHA256 == "" {
		return true
	}
	if sidecar, err := openRegularDownloadFile(root, sidecarPath(name), os.O_RDONLY); err == nil {
		var stored digestSidecar
		err := json.NewDecoder(io.LimitReader(sidecar, 4096)).Decode(&stored)
		sidecar.Close()
		if err == nil && stored.Size == info.Size() && stored.ModTime == info.ModTime().UnixNano() {
			return stored.SHA256 == f.SHA256
		}
	}
	h := sha256.New()
	if _, err := hashPrefix(root, h, name); err != nil || hex.EncodeToString(h.Sum(nil)) != f.SHA256 {
		return false
	}
	recordVerifiedDigest(root, name, f.SHA256)
	return true
}

// pruneCheckpointStage keeps the plan, its parent directories and checksum sidecars.
func pruneCheckpointStage(root *os.Root, files []File) error {
	regular := make(map[string]bool, len(files))
	directories := map[string]bool{".": true}
	for _, f := range files {
		regular[f.Path] = true
		if f.SHA256 != "" {
			regular[filepath.ToSlash(sidecarPath(filepath.FromSlash(f.Path)))] = true
		}
		for dir := path.Dir(f.Path); !directories[dir]; dir = path.Dir(dir) {
			directories[dir] = true
		}
	}
	return fs.WalkDir(root.FS(), ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if directories[name] && entry.IsDir() || regular[name] && entry.Type().IsRegular() {
			return nil
		}
		if err := root.RemoveAll(filepath.FromSlash(name)); err != nil {
			return err
		}
		if entry.IsDir() {
			return fs.SkipDir
		}
		return nil
	})
}

// publishCheckpoint claims the staging name inside a private directory and then checks its identity.
// Verification and cleanup use the same open directory that received the transfers.
func publishCheckpoint(parent, stage *os.Root, staging, dest string, p Plan, metadata checkpointMetadata) (err error) {
	info, err := parent.Lstat(staging)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(stage, info); err != nil {
		return err
	}
	work := ".goinfer-publish-" + rand.Text()
	if err := parent.Mkdir(work, 0o700); err != nil {
		return err
	}
	defer parent.Remove(work)
	claimed := filepath.Join(work, "checkpoint")
	if err := parent.Rename(staging, claimed); err != nil {
		return err
	}
	defer func() {
		if err == nil {
			return
		}
		if _, statErr := parent.Lstat(staging); os.IsNotExist(statErr) {
			if parent.Rename(claimed, staging) == nil {
				return
			}
		}
		err = errors.Join(err, fmt.Errorf("staging data remains in %s", filepath.Join(parent.Name(), claimed)))
	}()
	info, err = parent.Lstat(claimed)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(stage, info); err != nil {
		return err
	}
	if err := pruneCheckpointStage(stage, p.Files); err != nil {
		return fmt.Errorf("cleaning staged files: %w", err)
	}
	files, err := checkpointMarkerFiles(stage, p, metadata)
	if err != nil {
		return err
	}
	marker, err := json.MarshalIndent(markerDoc{Repo: p.Repo, Files: files}, "", " ")
	if err != nil {
		return err
	}
	if err := replaceRootFile(stage, checkpointMarker, marker); err != nil {
		return err
	}
	info, err = parent.Lstat(claimed)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(stage, info); err != nil {
		return err
	}
	if _, err := parent.Lstat(dest); !os.IsNotExist(err) {
		if err == nil {
			return fmt.Errorf("destination %s already exists", dest)
		}
		return err
	}
	return parent.Rename(claimed, dest)
}
