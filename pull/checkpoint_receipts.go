package pull

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const checkpointReceiptPrefix = ".goinfer-transfer-"

type checkpointReceipt struct {
	Owner      checkpointWorkspaceOwner      `json:"owner"`
	Path       string                        `json:"path"`
	Size       int64                         `json:"size"`
	ModTime    int64                         `json:"mtime"`
	SHA256     string                        `json:"sha256,omitempty"`
	Partial    bool                          `json:"partial,omitempty"`
	Completion *checkpointTransferCompletion `json:"completion,omitempty"`
	Identity   *checkpointTransferIdentity   `json:"identity,omitempty"`

	dirName string
	dirInfo os.FileInfo
}

type checkpointTransferCompletion struct {
	Size    int64  `json:"size"`
	ModTime int64  `json:"mtime"`
	SHA256  string `json:"sha256"`
}

type checkpointTransferIdentity struct {
	FileID     string `json:"file_id"`
	ChangeTime int64  `json:"change_time"`
}

type checkpointTransferSnapshot struct {
	Path       string `json:"path,omitempty"`
	Size       int64  `json:"size"`
	ModTime    int64  `json:"mtime"`
	FileID     string `json:"file_id,omitempty"`
	NoProof    bool   `json:"no_proof,omitempty"`
	ChangeTime int64  `json:"change_time,omitempty"`
	SHA256     string `json:"sha256,omitempty"`
}

// Record proof of the staged file so publication cleanup can be restarted.
func (w *checkpointWorkspace) recordTransfer(stage *os.Root, file File) {
	info, err := stage.Lstat(filepath.FromSlash(file.Path))
	if err != nil {
		return
	}
	w.recordCompletedTransfer(stage, file, info, file.SHA256)
}

func (w *checkpointWorkspace) recordCompletedTransfer(stage *os.Root, file File, info os.FileInfo, digest string) {
	if !info.Mode().IsRegular() || info.Size() != file.Size {
		return
	}
	receipt := checkpointReceipt{
		Owner: w.owner, Path: file.Path, Size: info.Size(), ModTime: info.ModTime().UnixNano(), SHA256: digest,
	}
	// Publication can use the streamed digest even when hard links are unavailable.
	if w.completed == nil {
		w.completed = make(map[string]checkpointReceipt)
	}
	w.completed[file.Path] = receipt
	w.recordReceipt(stage, receipt, info)
	w.pruneTransferReceipts(context.Background(), stage, file.Path, nil)
	w.pruneTransferReceipts(context.Background(), stage, file.Path+".part", nil)
}

var createCheckpointPartFile = openRegularDownloadFile
var linkCheckpointTransfer = (*os.Root).Link

// Create the inode at its recorded location first. Linking it into staging then exposes
// the canonical partial atomically, without replacing an existing file.
func (w *checkpointWorkspace) createPartial(stage *os.Root, name, digest string) (*os.File, string, error) {
	receipt := checkpointReceipt{Owner: w.owner, Path: filepath.ToSlash(name), Partial: true, SHA256: digest}
	r, err := w.newTransferReceipt(&receipt)
	if err != nil {
		return nil, "", err
	}
	defer r.Close()
	out, err := createCheckpointPartFile(r, "file", os.O_WRONLY|os.O_CREATE|os.O_EXCL)
	if err != nil {
		return nil, "", err
	}
	keep, linked := false, false
	defer func() {
		if !keep {
			out.Close()
			if !linked {
				w.removeTransferReceipt(r, receipt)
			}
		}
	}()
	if err := w.checkStageIdentity(stage); err != nil {
		return nil, "", err
	}
	private := filepath.Join(receipt.dirName, "file")
	if err := linkCheckpointTransfer(w.root, private, filepath.Join(checkpointStageName, name)); err != nil {
		if _, statErr := stage.Lstat(name); !os.IsNotExist(statErr) {
			return nil, "", err
		}
		// A filesystem without hard links can still pull through this private file.
		// The stage has no resumable name until the private file is renamed on success.
	} else {
		linked = true
		info, err := out.Stat()
		if err != nil {
			return nil, "", err
		}
		staged, err := stage.Lstat(name)
		if err != nil || !staged.Mode().IsRegular() || !os.SameFile(staged, info) {
			return nil, "", fmt.Errorf("partial %s changed while linking", name)
		}
		private = ""
	}
	w.pruneTransferReceipts(context.Background(), stage, receipt.Path, nil)
	w.addTransferReceipt(receipt)
	keep = true
	return out, private, nil
}

func (w *checkpointWorkspace) newTransferReceipt(receipt *checkpointReceipt) (*os.Root, error) {
	w.indexTransferReceipts()
	raw, err := json.Marshal(receipt)
	if err != nil {
		return nil, err
	}
	record := checkpointReceiptPrefix + rand.Text()
	if err := w.root.Mkdir(record, 0o700); err != nil {
		return nil, err
	}
	r, err := openCheckpointStaging(w.root, record)
	if err != nil {
		return nil, err
	}
	dirInfo, err := r.Stat(".")
	if err != nil {
		r.Close()
		return nil, err
	}
	if err := replaceRootFile(r, "transfer.json", raw); err != nil {
		r.Close()
		return nil, err
	}
	receipt.dirName, receipt.dirInfo = record, dirInfo
	return r, nil
}

func (w *checkpointWorkspace) recordReceipt(stage *os.Root, receipt checkpointReceipt, info os.FileInfo) {
	r, err := w.newTransferReceipt(&receipt)
	if err != nil {
		return
	}
	defer r.Close()
	if err := linkCheckpointTransfer(w.root, filepath.Join(checkpointStageName, filepath.FromSlash(receipt.Path)), filepath.Join(receipt.dirName, "file")); err != nil {
		if !w.recordTransferIdentity(stage, r, &receipt, info) {
			w.removeTransferRecord(r, receipt)
			return
		}
	} else if proof, err := r.Lstat("file"); err != nil || !proof.Mode().IsRegular() || !sameCheckpointFileState(proof, info) {
		w.removeTransferReceipt(r, receipt)
		return
	}
	w.addTransferReceipt(receipt)
}

func (w *checkpointWorkspace) addTransferReceipt(receipt checkpointReceipt) {
	w.receipts[receipt.Path] = append(w.receipts[receipt.Path], receipt)
}

// Record the verified bytes before rename, while the partial still names the recorded inode.
func (w *checkpointWorkspace) recordTransferCompletion(part string, info os.FileInfo, digest string) error {
	for i, receipt := range w.receipts[filepath.ToSlash(part)] {
		if !receipt.Partial {
			continue
		}
		r := w.openTransferReceipt(receipt)
		if r == nil {
			continue
		}
		proof, err := r.Lstat("file")
		if err != nil || !proof.Mode().IsRegular() || !sameCheckpointFileState(proof, info) {
			r.Close()
			continue
		}
		defer r.Close()
		receipt.Completion = &checkpointTransferCompletion{Size: info.Size(), ModTime: info.ModTime().UnixNano(), SHA256: digest}
		raw, err := json.Marshal(receipt)
		if err != nil {
			return err
		}
		if err := replaceRootFile(r, "transfer.json", raw); err != nil {
			return err
		}
		w.receipts[receipt.Path][i] = receipt
		return nil
	}
	return fmt.Errorf("partial %s has no verified transfer record at completion", part)
}

func (w *checkpointWorkspace) ownsPartial(name string, info os.FileInfo) bool {
	w.indexTransferReceipts()
	for _, receipt := range w.receipts[filepath.ToSlash(name)] {
		if receipt.Partial && w.matchesTransferProof(receipt, info) {
			return true
		}
	}
	return false
}

func (w *checkpointWorkspace) matchesTransferProof(receipt checkpointReceipt, info os.FileInfo) bool {
	r := w.openTransferReceipt(receipt)
	if r == nil {
		return false
	}
	defer r.Close()
	proof, err := r.Lstat("file")
	return err == nil && proof.Mode().IsRegular() && os.SameFile(proof, info)
}

func (w *checkpointWorkspace) snapshotTransfers(stage *os.Root) map[string]checkpointTransferSnapshot {
	w.indexTransferReceipts()
	result := make(map[string]checkpointTransferSnapshot)
	for _, receipts := range w.receipts {
		for _, receipt := range receipts {
			if state, ok := w.captureTransferSnapshot(stage, receipt); ok {
				result[receipt.dirName] = state
			}
		}
	}
	return result
}

func (w *checkpointWorkspace) captureTransferSnapshot(stage *os.Root, receipt checkpointReceipt) (checkpointTransferSnapshot, bool) {
	r := w.openTransferReceipt(receipt)
	if r == nil {
		return checkpointTransferSnapshot{}, false
	}
	defer r.Close()
	if receipt.Identity != nil {
		state, _, ok := receipt.identitySnapshot(stage, r)
		return state, ok && (state.NoProof || state == receipt.recordedSnapshot())
	}
	file, err := openRegularDownloadFile(r, "file", os.O_RDONLY)
	if os.IsNotExist(err) {
		return checkpointTransferSnapshot{NoProof: true}, true
	}
	if err != nil {
		return checkpointTransferSnapshot{}, false
	}
	defer file.Close()
	proof, err := file.Stat()
	if err != nil {
		return checkpointTransferSnapshot{}, false
	}
	path, _, err := transferStagePath(stage, receipt, proof)
	return checkpointTransferSnapshot{Path: filepath.ToSlash(path), Size: proof.Size(), ModTime: proof.ModTime().UnixNano(), FileID: checkpointFileID(file, proof)}, err == nil
}

// A later writer can reuse files recorded by the publication. Missing receipts
// are only consistent with cleanup when their stage files are also gone.
func (w *checkpointWorkspace) canResumeCleanup(ctx context.Context, stage *os.Root, sealed map[string]checkpointTransferSnapshot) bool {
	if len(sealed) == 0 {
		return false
	}
	w.indexTransferReceipts()
	seen := make(map[string]bool, len(sealed))
	for _, receipts := range w.receipts {
		for _, receipt := range receipts {
			previous, ok := sealed[receipt.dirName]
			if !ok || ctx.Err() != nil {
				return false
			}
			seen[receipt.dirName] = true
			current, ok := w.captureTransferSnapshot(stage, receipt)
			if !ok {
				return false
			}
			if (current.NoProof || current.Path == "") && !previous.stagePathGone(stage) {
				return false
			}
			if current.NoProof {
				continue // cleanup may already have removed the proof
			}
			if current.Path == "" {
				current.Path = previous.Path // cleanup may already have unlinked the staged entry
			}
			if current != previous {
				return false
			}
		}
	}
	for name, previous := range sealed {
		if !seen[name] && !previous.stagePathGone(stage) {
			return false
		}
	}
	return ctx.Err() == nil
}

func (s checkpointTransferSnapshot) stagePathGone(stage *os.Root) bool {
	if s.Path == "" {
		return true
	}
	_, err := stage.Lstat(filepath.FromSlash(s.Path))
	return os.IsNotExist(err)
}

func (w *checkpointWorkspace) reclaimTransfers(ctx context.Context, stage *os.Root, cleanup *checkpointCleanup) {
	if len(cleanup.Transfers) == 0 || ctx.Err() != nil {
		return
	}
	w.indexTransferReceipts()
	for path := range w.receipts {
		if ctx.Err() != nil {
			return
		}
		w.pruneTransferReceipts(ctx, stage, path, cleanup)
	}
}

func (w *checkpointWorkspace) pruneTransferReceipts(ctx context.Context, stage *os.Root, path string, cleanup *checkpointCleanup) {
	remaining := slices.DeleteFunc(w.receipts[path], func(receipt checkpointReceipt) bool {
		return w.releaseTransfer(ctx, stage, receipt, cleanup)
	})
	if len(remaining) > 0 {
		w.receipts[path] = remaining
	} else {
		delete(w.receipts, path)
	}
}

// The destination lock protects the index for the lifetime of this workspace handle.
func (w *checkpointWorkspace) indexTransferReceipts() {
	if w.receipts != nil {
		return
	}
	w.receipts = make(map[string][]checkpointReceipt)
	dir, err := w.root.Open(".")
	if err != nil {
		return
	}
	entries, err := dir.ReadDir(-1)
	dir.Close()
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), checkpointReceiptPrefix) {
			if receipt, ok := w.readTransferReceipt(entry.Name()); ok {
				w.receipts[receipt.Path] = append(w.receipts[receipt.Path], receipt)
			}
		}
	}
}

func (w *checkpointWorkspace) readTransferReceipt(name string) (checkpointReceipt, bool) {
	var receipt checkpointReceipt
	r, err := openCheckpointStaging(w.root, name)
	if err != nil {
		return receipt, false
	}
	defer r.Close()
	f, err := openRegularDownloadFile(r, "transfer.json", os.O_RDONLY)
	if err != nil {
		return receipt, false
	}
	err = json.NewDecoder(io.LimitReader(f, 8192)).Decode(&receipt)
	f.Close()
	if err != nil || receipt.Owner != w.owner || !safeRepoPath(receipt.Path) || !filepath.IsLocal(filepath.FromSlash(receipt.Path)) || receipt.Path == checkpointOwnerName || receipt.Path == checkpointPublicationName || receipt.Path == checkpointCleanupName {
		return receipt, false
	}
	if receipt.Identity != nil && (receipt.Partial || receipt.SHA256 == "" || receipt.Identity.FileID == "" || receipt.Identity.ChangeTime == 0) {
		return receipt, false
	}
	if receipt.Partial {
		final, ok := strings.CutSuffix(receipt.Path, ".part")
		if !ok || !safeRepoPath(final) || final == checkpointOwnerName || final == checkpointPublicationName || final == checkpointCleanupName {
			return receipt, false
		}
	}
	receipt.dirName = name
	receipt.dirInfo, err = r.Stat(".")
	return receipt, err == nil
}

func (w *checkpointWorkspace) openTransferReceipt(receipt checkpointReceipt) *os.Root {
	r, err := openCheckpointStaging(w.root, receipt.dirName)
	if err != nil {
		return nil
	}
	if !sameCheckpointDirectory(r, receipt.dirInfo) {
		r.Close()
		return nil
	}
	return r
}

func (w *checkpointWorkspace) releaseTransfer(ctx context.Context, stage *os.Root, receipt checkpointReceipt, cleanup *checkpointCleanup) bool {
	if ctx.Err() != nil {
		return false
	}
	var sealed *checkpointTransferSnapshot
	if cleanup != nil {
		state, ok := cleanup.Transfers[receipt.dirName]
		if !ok {
			return false
		}
		sealed = &state
	}
	r := w.openTransferReceipt(receipt)
	if r == nil {
		return false
	}
	defer r.Close()
	if receipt.Identity != nil {
		return w.releaseIdentityTransfer(ctx, stage, r, receipt, sealed, cleanup)
	}
	proof, err := r.Lstat("file")
	switch {
	case err == nil && proof.Mode().IsRegular():
		path, info, err := transferStagePath(stage, receipt, proof)
		if err != nil {
			return false
		}
		staged := path != ""
		renamedPartial := staged && receipt.Partial && path != filepath.FromSlash(receipt.Path)
		if sealed == nil && staged && (!renamedPartial || !w.hasCompletedTransfer(filepath.ToSlash(path), info)) {
			return false
		}
		if sealed != nil {
			if !sealed.matches(r, proof, path) {
				return false
			}
			if staged && receipt.canReclaim(ctx, stage, path, info, renamedPartial) {
				current, err := stage.Lstat(path)
				if err != nil || !sameCheckpointFileState(info, current) || ctx.Err() != nil || stage.Remove(path) != nil {
					return false
				}
				if (!receipt.Partial || renamedPartial) && receipt.SHA256 != "" {
					cleanup.removeStageSidecar(stage, path, info, receipt.SHA256)
				}
			}
		}
	case !os.IsNotExist(err):
		return false
	}
	return ctx.Err() == nil && w.removeTransferReceipt(r, receipt)
}

func transferStagePath(stage *os.Root, receipt checkpointReceipt, proof os.FileInfo) (string, os.FileInfo, error) {
	path := filepath.FromSlash(receipt.Path)
	info, err := stage.Lstat(path)
	if err == nil && info.Mode().IsRegular() && os.SameFile(info, proof) {
		return path, info, nil
	}
	if err != nil && !os.IsNotExist(err) {
		return "", nil, err
	}
	if receipt.Partial {
		final := strings.TrimSuffix(path, ".part")
		info, err := stage.Lstat(final)
		if err == nil && info.Mode().IsRegular() && os.SameFile(info, proof) {
			return final, info, nil
		}
		if err != nil && !os.IsNotExist(err) {
			return "", nil, err
		}
	}
	return "", nil, nil
}

func (s checkpointTransferSnapshot) matches(root *os.Root, proof os.FileInfo, path string) bool {
	if s.NoProof || (path != "" && filepath.ToSlash(path) != s.Path) || proof.Size() != s.Size || proof.ModTime().UnixNano() != s.ModTime {
		return false
	}
	if s.FileID == "" {
		return true
	}
	f, err := openRegularDownloadFile(root, "file", os.O_RDONLY)
	if err != nil {
		return false
	}
	defer f.Close()
	info, err := f.Stat()
	return err == nil && sameCheckpointFileState(proof, info) && checkpointFileID(f, info) == s.FileID
}

func (r checkpointReceipt) canReclaim(ctx context.Context, stage *os.Root, path string, info os.FileInfo, renamedPartial bool) bool {
	if r.Partial && !renamedPartial {
		return true
	}
	verified := checkpointTransferCompletion{Size: r.Size, ModTime: r.ModTime, SHA256: r.SHA256}
	if renamedPartial {
		// A pre-rename record also proves the contents if completion was interrupted.
		if r.Completion == nil {
			return false
		}
		verified = *r.Completion
	}
	if verified.SHA256 == "" || info.Size() != verified.Size || info.ModTime().UnixNano() != verified.ModTime {
		return false
	}
	f, err := openRegularDownloadFile(stage, path, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !sameCheckpointFileState(info, opened) {
		return false
	}
	return checkOpenCheckpointDigest(ctx, stage, path, File{Path: path, Size: verified.Size, SHA256: verified.SHA256}, f, info) == nil
}

func (w *checkpointWorkspace) hasCompletedTransfer(path string, info os.FileInfo) bool {
	for _, receipt := range w.receipts[path] {
		if !receipt.Partial && receipt.Size == info.Size() && receipt.ModTime == info.ModTime().UnixNano() && w.matchesTransferProof(receipt, info) {
			return true
		}
	}
	return false
}

func (w *checkpointWorkspace) removeTransferReceipt(r *os.Root, receipt checkpointReceipt) bool {
	if receipt.Identity == nil {
		if err := r.Remove("file"); err != nil && !os.IsNotExist(err) {
			return false
		}
	}
	return w.removeTransferRecord(r, receipt)
}

func (w *checkpointWorkspace) removeTransferRecord(r *os.Root, receipt checkpointReceipt) bool {
	if err := r.Remove("transfer.json"); err != nil && !os.IsNotExist(err) {
		return false
	}
	if info, err := w.root.Lstat(receipt.dirName); err == nil && sameCheckpointDirectory(r, info) {
		_ = w.root.Remove(receipt.dirName) // never remove an unrelated entry inside a record directory
	}
	return true
}
