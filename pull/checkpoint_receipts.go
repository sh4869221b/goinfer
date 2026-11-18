package pull

import (
	"crypto/rand"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
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
	w.retireSupersededTransfers(stage, file.Path, file.Path+".part")
}

// Only an exclusively created partial can be claimed. A failed transfer or
// SIGKILL can then leave bytes on the same inode without invalidating ownership.
func (w *checkpointWorkspace) recordPartial(stage *os.Root, name string, info os.FileInfo) {
	path := filepath.ToSlash(name)
	w.recordReceipt(checkpointReceipt{Owner: w.owner, Path: path, Partial: true}, info)
	w.retireSupersededTransfers(stage, path)
}

func (w *checkpointWorkspace) recordReceipt(receipt checkpointReceipt, info os.FileInfo) {
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
	}
}

func (w *checkpointWorkspace) reclaimTransfers(stage *os.Root) {
	w.visitTransferReceipts(func(r *os.Root, name string, receipt checkpointReceipt) {
		w.releaseTransfer(stage, r, name, receipt, true)
	})
}

func (w *checkpointWorkspace) retireSupersededTransfers(stage *os.Root, paths ...string) {
	w.visitTransferReceipts(func(r *os.Root, name string, receipt checkpointReceipt) {
		for _, path := range paths {
			if path == receipt.Path {
				w.releaseTransfer(stage, r, name, receipt, false)
				return
			}
		}
	})
}

func (w *checkpointWorkspace) visitTransferReceipts(visit func(*os.Root, string, checkpointReceipt)) {
	entries, err := fs.ReadDir(w.root.FS(), ".")
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), checkpointReceiptPrefix) {
			w.visitTransferReceipt(entry.Name(), visit)
		}
	}
}

func (w *checkpointWorkspace) visitTransferReceipt(name string, visit func(*os.Root, string, checkpointReceipt)) {
	r, err := openCheckpointStaging(w.root, name)
	if err != nil {
		return
	}
	defer r.Close()
	f, err := openRegularDownloadFile(r, "transfer.json", os.O_RDONLY)
	if err != nil {
		return
	}
	var receipt checkpointReceipt
	err = json.NewDecoder(io.LimitReader(f, 8192)).Decode(&receipt)
	f.Close()
	if err != nil || receipt.Owner != w.owner || !safeRepoPath(receipt.Path) || !filepath.IsLocal(filepath.FromSlash(receipt.Path)) || receipt.Path == checkpointOwnerName {
		return
	}
	if receipt.Partial && !strings.HasSuffix(receipt.Path, ".part") {
		return
	}
	visit(r, name, receipt)
}

func (w *checkpointWorkspace) releaseTransfer(stage, r *os.Root, name string, receipt checkpointReceipt, reclaim bool) {
	proof, err := r.Lstat("file")
	switch {
	case err == nil && proof.Mode().IsRegular():
		path := filepath.FromSlash(receipt.Path)
		info, err := stage.Lstat(path)
		if err != nil && !os.IsNotExist(err) {
			return
		}
		stillStaged := err == nil && info.Mode().IsRegular() && os.SameFile(info, proof)
		if stillStaged && !reclaim {
			return
		}
		if stillStaged && (receipt.Partial || info.Size() == receipt.Size && info.ModTime().UnixNano() == receipt.ModTime) {
			if err := stage.Remove(path); err != nil {
				return
			}
			if receipt.SHA256 != "" && checkpointDigestSidecar(stage, path, info) == receipt.SHA256 {
				_ = stage.Remove(sidecarPath(path))
			}
		}
		if err := r.Remove("file"); err != nil {
			return
		}
	case !os.IsNotExist(err):
		return
	}
	_ = r.Remove("transfer.json")
	_ = w.root.Remove(name) // never remove an unrelated entry inside a record directory
}
