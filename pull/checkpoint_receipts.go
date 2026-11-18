package pull

import (
	"crypto/rand"
	"encoding/json"
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
	name := filepath.FromSlash(file.Path)
	info, err := stage.Lstat(name)
	if err != nil || !info.Mode().IsRegular() || info.Size() != file.Size {
		return
	}
	w.recordReceipt(checkpointReceipt{
		Owner: w.owner, Path: file.Path, Size: info.Size(), ModTime: info.ModTime().UnixNano(), SHA256: file.SHA256,
	}, info)
	w.pruneTransferReceipts(stage, file.Path, false)
	w.pruneTransferReceipts(stage, file.Path+".part", false)
}

// Only an exclusively created partial can be claimed. A failed transfer or
// SIGKILL can then leave bytes on the same inode without invalidating ownership.
func (w *checkpointWorkspace) recordPartial(stage *os.Root, name string, info os.FileInfo) {
	path := filepath.ToSlash(name)
	w.recordReceipt(checkpointReceipt{Owner: w.owner, Path: path, Partial: true}, info)
	w.pruneTransferReceipts(stage, path, false)
}

func (w *checkpointWorkspace) recordReceipt(receipt checkpointReceipt, info os.FileInfo) {
	w.indexTransferReceipts()
	raw, err := json.Marshal(receipt)
	if err != nil {
		return
	}
	record := checkpointReceiptPrefix + rand.Text()
	if err := w.root.Mkdir(record, 0o700); err != nil {
		return
	}
	r, err := openCheckpointStaging(w.root, record)
	if err != nil {
		return
	}
	defer r.Close()
	dirInfo, err := r.Stat(".")
	if err != nil {
		return
	}
	if err := replaceRootFile(r, "transfer.json", raw); err != nil {
		return
	}
	if err := w.root.Link(filepath.Join(checkpointStageName, filepath.FromSlash(receipt.Path)), filepath.Join(record, "file")); err != nil {
		_ = r.Remove("transfer.json")
		_ = w.root.Remove(record)
		return
	}
	if proof, err := r.Lstat("file"); err != nil || !os.SameFile(proof, info) {
		_ = r.Remove("file")
		_ = r.Remove("transfer.json")
		_ = w.root.Remove(record)
		return
	}
	receipt.dirName, receipt.dirInfo = record, dirInfo
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
	if receipt.Partial && !strings.HasSuffix(receipt.Path, ".part") {
		return receipt, false
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
		info, err := stage.Lstat(path)
		if err != nil && !os.IsNotExist(err) {
			return false
		}
		stillStaged := err == nil && info.Mode().IsRegular() && os.SameFile(info, proof)
		if stillStaged && !reclaim {
			return false
		}
		if stillStaged && (receipt.Partial || info.Size() == receipt.Size && info.ModTime().UnixNano() == receipt.ModTime) {
			if err := stage.Remove(path); err != nil {
				return false
			}
			if receipt.SHA256 != "" && checkpointDigestSidecar(stage, path, info) == receipt.SHA256 {
				_ = stage.Remove(sidecarPath(path))
			}
		}
		if err := r.Remove("file"); err != nil {
			return false
		}
	case !os.IsNotExist(err):
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
