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
	checkpointStageName       = "stage"
	checkpointOwnerName       = ".goinfer-owner.json"
	checkpointWorkSuffix      = ".pull2"
	checkpointPublicationName = ".goinfer-publication.json"
)

func checkpointWorkspaceName(dest string) string { return "." + dest + checkpointWorkSuffix }

type checkpointWorkspace struct {
	root     *os.Root
	lock     *os.File
	name     string
	owner    checkpointWorkspaceOwner
	receipts map[string][]checkpointReceipt
}

type checkpointWorkspaceOwner struct {
	Destination string `json:"destination"`
	Token       string `json:"token"`
}

type checkpointPublication struct {
	Owner checkpointWorkspaceOwner `json:"owner"`
	Token string                   `json:"token"`
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
	return lockCheckpointWorkspace(parent, dest, true)
}

func lockCheckpointWorkspace(parent *os.Root, dest string, create bool) (*checkpointWorkspace, error) {
	name := checkpointWorkspaceName(dest)
	if create {
		if _, err := parent.Lstat(name); os.IsNotExist(err) {
			owner := checkpointWorkspaceOwner{Destination: dest, Token: rand.Text()}
			if err := initializeCheckpointDirectory(parent, name, owner); err != nil && !os.IsExist(err) {
				return nil, err
			}
		}
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
	owner, err := readCheckpointOwner(workspace)
	if err != nil || owner.Destination != dest || owner.Token == "" {
		return nil, fmt.Errorf("workspace %s has no verified owner for %s; existing data was kept", name, dest)
	}
	flags := os.O_RDWR
	if create {
		flags |= os.O_CREATE
	}
	lock, err := openRegularDownloadFile(workspace, "lock", flags)
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

// A finished destination can be reused while its writer is still cleaning up.
// Retry interrupted cleanup only when the existing workspace can be locked.
func recoverCheckpointTransfers(dest string, complete *completedCheckpoint) {
	if complete.publication == "" {
		return
	}
	parent, err := os.OpenRoot(filepath.Dir(filepath.Clean(dest)))
	if err != nil {
		return
	}
	defer parent.Close()
	name := filepath.Base(dest)
	work, err := lockCheckpointWorkspace(parent, name, false)
	if err != nil {
		return
	}
	defer work.Close()
	work.reclaimCompletedStage(parent, name, complete)
}

func (w *checkpointWorkspace) reclaimCompletedStage(parent *os.Root, dest string, complete *completedCheckpoint) {
	info, err := parent.Stat(dest)
	if err != nil || !os.SameFile(info, complete.directory) {
		return
	}
	stage, err := w.openStage(false)
	if err != nil || stage == nil {
		return
	}
	defer stage.Close()
	if w.matchesPublication(stage, complete.publication) {
		w.reclaimTransfers(stage)
	}
}

// Rotate the stage token before mutating any files. A restored checkpoint from an earlier
// publication cannot authorize cleanup of a later transfer at the same destination.
func (w *checkpointWorkspace) beginPublication(stage *os.Root) (string, error) {
	var old checkpointPublication
	err := readCheckpointWorkspaceRecord(stage, checkpointPublicationName, &old)
	if !os.IsNotExist(err) && (err != nil || old.Owner != w.owner || old.Token == "") {
		return "", fmt.Errorf("%s has no verified owner", checkpointPublicationName)
	}
	next := checkpointPublication{Owner: w.owner, Token: rand.Text()}
	raw, err := json.Marshal(next)
	if err != nil {
		return "", err
	}
	return next.Token, replaceRootFile(stage, checkpointPublicationName, raw)
}

func (w *checkpointWorkspace) matchesPublication(stage *os.Root, token string) bool {
	var record checkpointPublication
	return token != "" && readCheckpointWorkspaceRecord(stage, checkpointPublicationName, &record) == nil && record.Owner == w.owner && record.Token == token
}

// Keep the stable workspace or stage name absent until its owner is fully recorded.
// An interrupted initializer can leave a temporary name, which is never resumed.
func initializeCheckpointDirectory(parent *os.Root, name string, owner checkpointWorkspaceOwner) error {
	tmp := ".goinfer-init-" + rand.Text()
	if err := mkdirCheckpointInit(parent, tmp); err != nil {
		return err
	}
	root, err := openCheckpointStaging(parent, tmp)
	if err != nil {
		return err
	}
	defer func() {
		if info, err := parent.Lstat(tmp); err == nil && sameCheckpointDirectory(root, info) {
			_ = root.Remove(checkpointOwnerName)
			_ = parent.Remove(tmp)
		}
		root.Close()
	}()
	if err := writeCheckpointOwner(root, owner); err != nil {
		return err
	}
	return renameCheckpointInit(parent, tmp, name)
}

// Indirected so the process test can stop at the mkdir boundary.
var mkdirCheckpointInit = func(root *os.Root, name string) error { return root.Mkdir(name, 0o700) }

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
	err := readCheckpointWorkspaceRecord(root, checkpointOwnerName, &owner)
	return owner, err
}

func readCheckpointWorkspaceRecord(root *os.Root, name string, value any) error {
	file, err := openRegularDownloadFile(root, name, os.O_RDONLY)
	if err != nil {
		return err
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 4097))
	if err != nil {
		return err
	}
	if len(raw) > 4096 {
		return fmt.Errorf("checkpoint workspace record %s is too large", name)
	}
	return json.Unmarshal(raw, value)
}

func (w *checkpointWorkspace) openStage(create bool) (*os.Root, error) {
	if _, err := w.root.Lstat(checkpointStageName); os.IsNotExist(err) {
		if !create {
			return nil, nil
		}
		if err := initializeCheckpointDirectory(w.root, checkpointStageName, w.owner); err != nil && !os.IsExist(err) {
			return nil, err
		}
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

func (w *checkpointWorkspace) checkStageIdentity(stage *os.Root) error {
	info, err := w.root.Lstat(checkpointStageName)
	if err != nil {
		return err
	}
	return checkCheckpointStaging(stage, info)
}

func reusableStagedFile(root *os.Root, name string, f File) bool {
	if f.SHA256 == "" {
		return false
	}
	opened, err := openRegularDownloadFile(root, name, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer opened.Close()
	info, err := opened.Stat()
	return err == nil && info.Size() == f.Size && verifyOpenCheckpointDigest(root, name, f, opened, info) == nil
}

func hashPublishedFile(root *os.Root, f File) (string, error) {
	name := filepath.FromSlash(f.Path)
	file, err := openRegularDownloadFile(root, name, os.O_RDONLY)
	if err != nil {
		return "", err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	digest, err := hashCheckpointFile(f, file)
	if err != nil {
		return "", err
	}
	if err := confirmCheckpointFile(root, name, file, info); err != nil {
		return "", err
	}
	writeRootDigest(root, name, info, checkpointFileID(file, info), digest)
	return digest, nil
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
func publishCheckpoint(parent *os.Root, work *checkpointWorkspace, stage *os.Root, dest string, p Plan, metadata checkpointMetadata, publication string) error {
	workspace, err := parent.Lstat(work.name)
	if err != nil {
		return err
	}
	if err := checkCheckpointStaging(work.root, workspace); err != nil {
		return err
	}
	if err := work.checkStageIdentity(stage); err != nil {
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
	marker, err := json.MarshalIndent(markerDoc{Repo: p.Repo, Files: files, Publication: publication}, "", " ")
	if err != nil {
		return err
	}
	if err := replaceRootFile(publish, checkpointMarker, marker); err != nil {
		return err
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
	if err := renameCheckpointPublication(parent, relativePublish, dest); err != nil {
		return err
	}
	if !work.matchesPublication(stage, publication) {
		return nil
	}
	// Remove only the source files that were published. Unlisted data in the
	// workspace may belong to someone else, even when the stage belongs to us.
	for _, file := range files {
		name := filepath.FromSlash(file.Path)
		if info, err := stage.Lstat(name); err == nil && sameCheckpointFileState(sources[file.Path], info) {
			if err := stage.Remove(name); err == nil && file.SHA256 != "" && checkpointDigestSidecar(stage, name, info) == file.SHA256 {
				_ = stage.Remove(sidecarPath(name))
			}
		}
	}
	work.reclaimTransfers(stage)
	return nil
}

var renameCheckpointPublication = (*os.Root).Rename

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
	if f.SHA256 != "" {
		if err := verifyOpenCheckpointDigest(stage, name, f, in, source); err != nil {
			return nil, err
		}
	}
	if err := publish.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		return nil, err
	}
	if err := work.Link(filepath.Join(checkpointStageName, name), filepath.Join(publishName, name)); err == nil {
		linked, err := publish.Lstat(name)
		if err != nil {
			return nil, err
		}
		if !linked.Mode().IsRegular() || !sameCheckpointFileState(source, linked) {
			return nil, fmt.Errorf("staged file changed while linking")
		}
		if f.SHA256 != "" {
			writeRootDigest(publish, name, linked, checkpointFileID(in, source), f.SHA256)
		}
		return source, nil
	}
	// Filesystems without hard links can publish via a copy on the same disk.
	if err := checkDiskSpace(f.Path, f.Size, publish.Name()); err != nil {
		return nil, err
	}
	if _, err := in.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	out, err := openRegularDownloadFile(publish, name, os.O_CREATE|os.O_EXCL|os.O_WRONLY)
	if err != nil {
		return nil, err
	}
	var copyErr error
	if f.SHA256 != "" {
		digest, err := hashCheckpointFile(f, io.TeeReader(in, out))
		copyErr = err
		if err == nil && digest != f.SHA256 {
			copyErr = fmt.Errorf("copied file does not match the checkpoint digest")
		}
		if copyErr == nil {
			info, err := out.Stat()
			copyErr = err
			if err == nil {
				writeRootDigest(publish, name, info, checkpointFileID(out, info), digest)
			}
		}
	} else {
		var n int64
		n, copyErr = io.Copy(out, in)
		if copyErr == nil && n != f.Size {
			copyErr = fmt.Errorf("staged file changed while copying")
		}
	}
	closeErr := out.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return source, nil
}
