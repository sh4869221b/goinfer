package pull

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	checkpointStageName  = "stage"
	checkpointOwnerName  = ".goinfer-owner.json"
	checkpointWorkSuffix = ".pull2"
)

func checkpointWorkspaceName(dest string) string { return "." + dest + checkpointWorkSuffix }

type checkpointWorkspace struct {
	root  *os.Root
	lock  *os.File
	name  string
	owner checkpointWorkspaceOwner
}

type checkpointWorkspaceOwner struct {
	Destination string `json:"destination"`
	Token       string `json:"token"`
}

func (w *checkpointWorkspace) Close() {
	w.lock.Close()
	w.root.Close()
}

func validateCheckpointDestination(dest string) error {
	abs, err := filepath.Abs(dest)
	if err != nil {
		return err
	}
	resolved, _ := filepath.EvalSymlinks(nearestDir(filepath.Dir(abs)))
	for _, path := range []string{abs, resolved} {
		for name := range strings.SplitSeq(filepath.ToSlash(filepath.Clean(path)), "/") {
			if strings.HasPrefix(name, ".") && strings.HasSuffix(strings.ToLower(name), checkpointWorkSuffix) {
				return fmt.Errorf("%s is inside a checkpoint workspace", dest)
			}
		}
	}
	return nil
}

// A lock file stays in place so all writers use the same inode. Closing its handle
// (including on process death) releases the lock. The claim belongs to the same destination.
func lockCheckpointDestination(parent *os.Root, dest string) (*checkpointWorkspace, error) {
	name := checkpointWorkspaceName(dest)
	mkdirErr := parent.Mkdir(name, 0o700)
	if mkdirErr != nil && !os.IsExist(mkdirErr) {
		return nil, mkdirErr
	}
	workspace, err := openCheckpointStaging(parent, name)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			workspace.Close()
		}
	}()
	owner := checkpointWorkspaceOwner{Destination: dest, Token: rand.Text()}
	if mkdirErr == nil {
		if err := writeCheckpointOwner(workspace, owner); err != nil {
			return nil, err
		}
	} else {
		owner, err = readCheckpointOwner(workspace)
		if err != nil || owner.Destination != dest || owner.Token == "" {
			return nil, fmt.Errorf("workspace %s has no verified owner for %s; existing data was kept", name, dest)
		}
	}
	lock, err := openRegularDownloadFile(workspace, "lock", os.O_RDWR|os.O_CREATE)
	if err != nil {
		return nil, err
	}
	acquired, err := tryCheckpointLock(lock)
	if err != nil || !acquired {
		lock.Close()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("another checkpoint pull is running for %s", dest)
	}
	ok = true
	return &checkpointWorkspace{root: workspace, lock: lock, name: name, owner: owner}, nil
}

func writeCheckpointOwner(root *os.Root, owner checkpointWorkspaceOwner) error {
	raw, err := json.Marshal(owner)
	if err != nil {
		return err
	}
	file, err := openRegularDownloadFile(root, checkpointOwnerName, os.O_CREATE|os.O_EXCL|os.O_WRONLY)
	if err != nil {
		return err
	}
	_, err = file.Write(raw)
	if closeErr := file.Close(); err == nil {
		return closeErr
	}
	return err
}

func readCheckpointOwner(root *os.Root) (checkpointWorkspaceOwner, error) {
	var owner checkpointWorkspaceOwner
	file, err := openRegularDownloadFile(root, checkpointOwnerName, os.O_RDONLY)
	if err != nil {
		return owner, err
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 4097))
	if err != nil {
		return owner, err
	}
	if len(raw) > 4096 {
		return owner, fmt.Errorf("checkpoint owner record is too large")
	}
	err = json.Unmarshal(raw, &owner)
	return owner, err
}

func (w *checkpointWorkspace) openStage(create bool) (*os.Root, error) {
	if _, err := w.root.Lstat(checkpointStageName); os.IsNotExist(err) {
		if !create {
			return nil, nil
		}
		if err := w.root.Mkdir(checkpointStageName, 0o700); err != nil {
			return nil, err
		}
		stage, err := openCheckpointStaging(w.root, checkpointStageName)
		if err != nil {
			return nil, err
		}
		if err := writeCheckpointOwner(stage, w.owner); err != nil {
			stage.Close()
			return nil, err
		}
		return stage, nil
	}
	stage, err := openCheckpointStaging(w.root, checkpointStageName)
	if err != nil {
		return nil, err
	}
	owner, err := readCheckpointOwner(stage)
	if err != nil || owner != w.owner {
		stage.Close()
		return nil, fmt.Errorf("staging directory has no verified owner for %s; existing data was kept", w.owner.Destination)
	}
	return stage, nil
}

func reusableStagedFile(root *os.Root, name string, f File) bool {
	if f.SHA256 == "" {
		return false
	}
	info, err := root.Lstat(name)
	if err != nil || !info.Mode().IsRegular() || info.Size() != f.Size {
		return false
	}
	if digest := checkpointDigestSidecar(root, name, info); digest != "" {
		return digest == f.SHA256
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

// publishCheckpoint builds a clean directory from the plan. Staging keeps its
// identity, including when the writer is killed before the final rename.
func publishCheckpoint(parent *os.Root, work *checkpointWorkspace, stage *os.Root, dest string, p Plan, metadata checkpointMetadata) error {
	workspace, err := parent.Lstat(work.name)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(work.root, workspace); err != nil {
		return err
	}
	stageInfo, err := work.root.Lstat(checkpointStageName)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(stage, stageInfo); err != nil {
		return err
	}
	name := "publish-" + rand.Text()
	if err := work.root.Mkdir(name, 0o755); err != nil {
		return err
	}
	publish, err := openCheckpointStaging(work.root, name)
	if err != nil {
		return err
	}
	defer publish.Close()
	defer func() {
		if info, err := work.root.Lstat(name); err == nil && sameCheckpointDirectory(publish, info) {
			_ = work.root.RemoveAll(name)
		}
	}()
	sources := make(map[string]os.FileInfo, len(p.Files))
	for _, f := range p.Files {
		info, err := preparePublishedFile(work.root, stage, publish, name, f)
		if err != nil {
			return fmt.Errorf("publishing %s: %w", f.Path, err)
		}
		sources[f.Path] = info
	}
	files, err := checkpointMarkerFiles(publish, p, metadata)
	if err != nil {
		return err
	}
	marker, err := json.MarshalIndent(markerDoc{Repo: p.Repo, Files: files}, "", " ")
	if err != nil {
		return err
	}
	if err := replaceRootFile(publish, checkpointMarker, marker); err != nil {
		return err
	}
	for _, f := range files {
		recordVerifiedDigest(publish, filepath.FromSlash(f.Path), f.SHA256)
	}
	relativePublish := filepath.Join(work.name, name)
	ready, err := parent.Lstat(relativePublish)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(publish, ready); err != nil {
		return err
	}
	if _, err := parent.Lstat(dest); !os.IsNotExist(err) {
		if err == nil {
			return fmt.Errorf("destination %s already exists", dest)
		}
		return err
	}
	if err := parent.Rename(relativePublish, dest); err != nil {
		return err
	}
	// Remove only the source files that were published. Unlisted data in the
	// workspace may belong to someone else, even when the stage belongs to us.
	for _, file := range files {
		name := filepath.FromSlash(file.Path)
		if info, err := stage.Lstat(name); err == nil && os.SameFile(sources[file.Path], info) {
			if err := stage.Remove(name); err == nil && file.SHA256 != "" && checkpointDigestSidecar(stage, name, info) == file.SHA256 {
				_ = stage.Remove(sidecarPath(name))
			}
		}
	}
	return nil
}

func preparePublishedFile(work, stage, publish *os.Root, publishName string, f File) (os.FileInfo, error) {
	name := filepath.FromSlash(f.Path)
	in, err := openRegularDownloadFile(stage, name, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	source, err := in.Stat()
	if err != nil {
		return nil, err
	}
	if source.Size() != f.Size {
		return nil, fmt.Errorf("staged file has size %d, want %d", source.Size(), f.Size)
	}
	if f.SHA256 != "" && !reusableStagedFile(stage, name, f) {
		return nil, fmt.Errorf("staged file does not match the checkpoint digest")
	}
	if err := publish.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		return nil, err
	}
	if err := work.Link(filepath.Join(checkpointStageName, name), filepath.Join(publishName, name)); err == nil {
		linked, err := publish.Lstat(name)
		if err != nil {
			return nil, err
		}
		if !linked.Mode().IsRegular() || !os.SameFile(source, linked) {
			return nil, fmt.Errorf("staged file changed while linking")
		}
		return source, nil
	}
	// Filesystems without hard links can publish via a copy on the same disk.
	if err := checkDiskSpace(f.Path, f.Size, publish.Name()); err != nil {
		return nil, err
	}
	out, err := openRegularDownloadFile(publish, name, os.O_CREATE|os.O_EXCL|os.O_WRONLY)
	if err != nil {
		return nil, err
	}
	n, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if n != f.Size {
		return nil, fmt.Errorf("staged file changed while copying")
	}
	return source, nil
}
