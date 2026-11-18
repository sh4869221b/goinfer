package pull

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
)

// A completed transfer can be recorded without a hard link if the staged file
// provides both a file ID and a change time. Partial transfers still need a link.
func (w *checkpointWorkspace) recordTransferIdentity(stage, record *os.Root, receipt *checkpointReceipt, verified os.FileInfo) bool {
	if receipt.Partial || receipt.SHA256 == "" || w.checkStageIdentity(stage) != nil {
		return false
	}
	if _, err := record.Lstat("file"); !os.IsNotExist(err) {
		return false
	}
	f, err := openRegularDownloadFile(stage, filepath.FromSlash(receipt.Path), os.O_RDONLY)
	if err != nil {
		return false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !sameCheckpointFileState(info, verified) {
		return false
	}
	id, changed := checkpointFileID(f, info), checkpointChangeTime(info)
	if id == "" || changed == 0 {
		return false
	}
	receipt.Identity = &checkpointTransferIdentity{FileID: id, ChangeTime: changed}
	raw, err := json.Marshal(receipt)
	return err == nil && replaceRootFile(record, "transfer.json", raw) == nil
}

func (r checkpointReceipt) recordedSnapshot() checkpointTransferSnapshot {
	return checkpointTransferSnapshot{
		Path: r.Path, Size: r.Size, ModTime: r.ModTime, FileID: r.Identity.FileID,
		ChangeTime: r.Identity.ChangeTime, SHA256: r.SHA256,
	}
}

func (r checkpointReceipt) identitySnapshot(stage, record *os.Root) (checkpointTransferSnapshot, os.FileInfo, bool) {
	if _, err := record.Lstat("file"); !os.IsNotExist(err) {
		return checkpointTransferSnapshot{}, nil, false
	}
	f, err := openRegularDownloadFile(stage, filepath.FromSlash(r.Path), os.O_RDONLY)
	if os.IsNotExist(err) {
		return checkpointTransferSnapshot{NoProof: true}, nil, true
	}
	if err != nil {
		return checkpointTransferSnapshot{}, nil, false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return checkpointTransferSnapshot{}, nil, false
	}
	return checkpointTransferSnapshot{
		Path: r.Path, Size: info.Size(), ModTime: info.ModTime().UnixNano(), FileID: checkpointFileID(f, info),
		ChangeTime: checkpointChangeTime(info), SHA256: r.SHA256,
	}, info, true
}

func (w *checkpointWorkspace) releaseIdentityTransfer(ctx context.Context, stage, record *os.Root, receipt checkpointReceipt, sealed *checkpointTransferSnapshot, cleanup *checkpointCleanup) bool {
	state, info, ok := receipt.identitySnapshot(stage, record)
	if !ok {
		return false
	}
	if state.NoProof || sealed == nil && state != receipt.recordedSnapshot() {
		return ctx.Err() == nil && w.removeTransferReceipt(record, receipt)
	}
	if sealed == nil || state != *sealed || state != receipt.recordedSnapshot() {
		return false
	}
	path := filepath.FromSlash(receipt.Path)
	if !receipt.canReclaim(ctx, stage, path, info, false) {
		return false
	}
	current, err := stage.Lstat(path)
	if err != nil || !sameCheckpointFileState(info, current) || checkpointChangeTime(current) != state.ChangeTime || ctx.Err() != nil {
		return false
	}
	if err := stage.Remove(path); err != nil {
		return false
	}
	cleanup.removeStageSidecar(stage, path, info, receipt.SHA256)
	return w.removeTransferReceipt(record, receipt)
}
