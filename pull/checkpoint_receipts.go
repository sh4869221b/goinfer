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
}

// Save a hard link as proof of the transferred inode. A missing receipt or
// unsupported hard links means the file will be left in staging.
func (w *checkpointWorkspace) recordTransfer(stage *os.Root, file File) {
	name := filepath.FromSlash(file.Path)
	info, err := stage.Lstat(name)
	if err != nil || !info.Mode().IsRegular() || info.Size() != file.Size {
		return
	}
	receipt, err := json.Marshal(checkpointReceipt{w.owner, file.Path, info.Size(), info.ModTime().UnixNano(), file.SHA256})
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
	if err := replaceRootFile(r, "transfer.json", receipt); err != nil {
		return
	}
	if err := w.root.Link(filepath.Join(checkpointStageName, name), filepath.Join(record, "file")); err != nil {
		_ = r.Remove("transfer.json")
		_ = w.root.Remove(record)
		return
	}
	if proof, err := r.Lstat("file"); err != nil || !os.SameFile(proof, info) {
		_ = r.Remove("file")
	}
}

func (w *checkpointWorkspace) reclaimTransfers(stage *os.Root) {
	entries, err := fs.ReadDir(w.root.FS(), ".")
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), checkpointReceiptPrefix) {
			w.reclaimTransfer(stage, entry.Name())
		}
	}
}

func (w *checkpointWorkspace) reclaimTransfer(stage *os.Root, name string) {
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
	proof, err := r.Lstat("file")
	switch {
	case err == nil && proof.Mode().IsRegular():
		path := filepath.FromSlash(receipt.Path)
		if info, err := stage.Lstat(path); err == nil && info.Mode().IsRegular() && os.SameFile(info, proof) && info.Size() == receipt.Size && info.ModTime().UnixNano() == receipt.ModTime {
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
