package pull

import (
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
	Owner   checkpointWorkspaceOwner `json:"owner"`
	Path    string                   `json:"path"`
	Size    int64                    `json:"size"`
	ModTime int64                    `json:"mtime"`
	SHA256  string                   `json:"sha256,omitempty"`
	Partial bool                     `json:"partial,omitempty"`

	dirName string
	dirInfo os.FileInfo
}

// Save a hard link as proof of the transferred inode. A missing receipt or
// unsupported hard links means the file will be left in staging.
func (w *checkpointWorkspace) recordTransfer(stage *os.Root, file File) {
	info, err := stage.Lstat(filepath.FromSlash(file.Path))
	if err != nil {
		return
	}
	w.recordCompletedTransfer(stage, file, info)
}

func (w *checkpointWorkspace) recordCompletedTransfer(stage *os.Root, file File, info os.FileInfo) {
	if !info.Mode().IsRegular() || info.Size() != file.Size {
		return
	}
	w.recordReceipt(checkpointReceipt{
		Owner: w.owner, Path: file.Path, Size: info.Size(), ModTime: info.ModTime().UnixNano(), SHA256: file.SHA256,
	}, info)
	w.pruneTransferReceipts(stage, file.Path, false)
	w.pruneTransferReceipts(stage, file.Path+".part", false)
}

var createCheckpointPartFile = openRegularDownloadFile
var linkCheckpointPartial = (*os.Root).Link

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
	if err := linkCheckpointPartial(w.root, private, filepath.Join(checkpointStageName, name)); err != nil {
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
	w.pruneTransferReceipts(stage, receipt.Path, false)
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

func (w *checkpointWorkspace) recordReceipt(receipt checkpointReceipt, info os.FileInfo) {
	r, err := w.newTransferReceipt(&receipt)
	if err != nil {
		return
	}
	defer r.Close()
	if err := w.root.Link(filepath.Join(checkpointStageName, filepath.FromSlash(receipt.Path)), filepath.Join(receipt.dirName, "file")); err != nil {
		w.removeTransferReceipt(r, receipt)
		return
	}
	if proof, err := r.Lstat("file"); err != nil || !proof.Mode().IsRegular() || !sameCheckpointFileState(proof, info) {
		w.removeTransferReceipt(r, receipt)
		return
	}
	w.addTransferReceipt(receipt)
}

func (w *checkpointWorkspace) addTransferReceipt(receipt checkpointReceipt) {
	w.receipts[receipt.Path] = append(w.receipts[receipt.Path], receipt)
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

func (w *checkpointWorkspace) reclaimTransfers(stage *os.Root) {
	w.indexTransferReceipts()
	for path := range w.receipts {
		w.pruneTransferReceipts(stage, path, true)
	}
}

func (w *checkpointWorkspace) pruneTransferReceipts(stage *os.Root, path string, reclaim bool) {
	remaining := slices.DeleteFunc(w.receipts[path], func(receipt checkpointReceipt) bool {
		return w.releaseTransfer(stage, receipt, reclaim)
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
	if err != nil || receipt.Owner != w.owner || !safeRepoPath(receipt.Path) || !filepath.IsLocal(filepath.FromSlash(receipt.Path)) || receipt.Path == checkpointOwnerName {
		return receipt, false
	}
	if receipt.Partial {
		final, ok := strings.CutSuffix(receipt.Path, ".part")
		if !ok || !safeRepoPath(final) || final == checkpointOwnerName || final == checkpointPublicationName {
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

func (w *checkpointWorkspace) releaseTransfer(stage *os.Root, receipt checkpointReceipt, reclaim bool) bool {
	r := w.openTransferReceipt(receipt)
	if r == nil {
		return false
	}
	defer r.Close()
	proof, err := r.Lstat("file")
	switch {
	case err == nil && proof.Mode().IsRegular():
		path := filepath.FromSlash(receipt.Path)
		sidecar := !receipt.Partial
		info, err := stage.Lstat(path)
		if err != nil && !os.IsNotExist(err) {
			return false
		}
		stillStaged := err == nil && info.Mode().IsRegular() && os.SameFile(info, proof)
		if receipt.Partial && !stillStaged {
			// A writer can die after renaming the partial but before recording the final file.
			final := strings.TrimSuffix(path, ".part")
			completed, err := stage.Lstat(final)
			if err != nil && !os.IsNotExist(err) {
				return false
			}
			if err == nil && completed.Mode().IsRegular() && os.SameFile(completed, proof) {
				if !reclaim && !w.hasCompletedTransfer(filepath.ToSlash(final), completed) {
					return false
				}
				path, info, stillStaged = final, completed, reclaim
				sidecar = true
			}
		}
		if stillStaged && !reclaim {
			return false
		}
		if stillStaged && (receipt.Partial || info.Size() == receipt.Size && info.ModTime().UnixNano() == receipt.ModTime) {
			if err := stage.Remove(path); err != nil {
				return false
			}
			if sidecar && receipt.SHA256 != "" && checkpointDigestSidecar(stage, path, info) == receipt.SHA256 {
				_ = stage.Remove(sidecarPath(path))
			}
		}
	case !os.IsNotExist(err):
		return false
	}
	return w.removeTransferReceipt(r, receipt)
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
	if err := r.Remove("file"); err != nil && !os.IsNotExist(err) {
		return false
	}
	if err := r.Remove("transfer.json"); err != nil && !os.IsNotExist(err) {
		return false
	}
	if info, err := w.root.Lstat(receipt.dirName); err == nil && sameCheckpointDirectory(r, info) {
		_ = w.root.Remove(receipt.dirName) // never remove an unrelated entry inside a record directory
	}
	return true
}
