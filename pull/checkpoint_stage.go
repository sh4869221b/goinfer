package pull

import (
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
	"strings"
)

func checkpointWorkspaceName(dest string) string { return "." + dest + ".pull" }

// A lock file stays in place so all writers use the same inode. Closing its handle
// (including on process death) releases the lock. The claim belongs to the same destination.
func lockCheckpointDestination(parent *os.Root, dest string) (*os.File, string, error) {
	if strings.HasPrefix(dest, ".") && strings.HasSuffix(strings.ToLower(dest), ".pull") {
		return nil, "", fmt.Errorf("%s is reserved for checkpoint downloads", dest)
	}
	name := checkpointWorkspaceName(dest)
	if err := parent.Mkdir(name, 0o700); err != nil && !os.IsExist(err) {
		return nil, "", err
	}
	workspace, err := openCheckpointStaging(parent, name)
	if err != nil {
		return nil, "", err
	}
	defer workspace.Close()
	lock, err := openRegularDownloadFile(workspace, "lock", os.O_RDWR|os.O_CREATE)
	if err != nil {
		return nil, "", err
	}
	acquired, err := tryCheckpointLock(lock)
	if err != nil || !acquired {
		lock.Close()
		if err != nil {
			return nil, "", err
		}
		return nil, "", fmt.Errorf("another checkpoint pull is running for %s", dest)
	}
	return lock, filepath.Join(name, "checkpoint"), nil
}

func recoverCheckpointStage(parent *os.Root, staging, claim string) error {
	info, err := parent.Lstat(claim)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("checkpoint recovery path %s must be a directory", claim)
	}
	if _, err := parent.Lstat(staging); !os.IsNotExist(err) {
		if err != nil {
			return err
		}
		return fmt.Errorf("both %s and %s contain staging data; use a different output directory to keep both", staging, claim)
	}
	return parent.Rename(claim, staging)
}

func reusableStagedFile(root *os.Root, name string, f File) bool {
	if f.SHA256 == "" {
		return false
	}
	info, err := root.Lstat(name)
	if err != nil || !info.Mode().IsRegular() || info.Size() != f.Size {
		return false
	}
	if sidecar, err := openRegularDownloadFile(root, sidecarPath(name), os.O_RDONLY); err == nil {
		var stored digestSidecar
		err := json.NewDecoder(io.LimitReader(sidecar, 4096)).Decode(&stored)
		sidecar.Close()
		if err == nil && stored.Size == info.Size() && stored.ModTime == info.ModTime().UnixNano() {
			return stored.SHA256 == f.SHA256
		}
	}
	digest, err := hashStagedFile(root, f)
	if err != nil || digest != f.SHA256 {
		return false
	}
	recordVerifiedDigest(root, name, f.SHA256)
	return true
}

func hashStagedFile(root *os.Root, f File) (string, error) {
	file, err := openRegularDownloadFile(root, filepath.FromSlash(f.Path), os.O_RDONLY)
	if err != nil {
		return "", err
	}
	defer file.Close()
	return hashCheckpointFile(f, file)
}

func hashCheckpointFile(f File, r io.Reader) (string, error) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		return "", err
	}
	if n != f.Size {
		return "", fmt.Errorf("%s changed size during verification", f.Path)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
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
func publishCheckpoint(parent, stage *os.Root, staging, dest, claimed string, p Plan, metadata checkpointMetadata) (err error) {
	info, err := parent.Lstat(staging)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(stage, info); err != nil {
		return err
	}
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
