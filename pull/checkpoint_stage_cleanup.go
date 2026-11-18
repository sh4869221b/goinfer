package pull

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

type checkpointStageSidecar struct {
	FileID     string `json:"file_id"`
	ChangeTime int64  `json:"change_time,omitempty"`
	SHA256     string `json:"sha256"`
}

type checkpointStageEntry struct {
	Mode       fs.FileMode `json:"mode"`
	FileID     string      `json:"file_id,omitempty"`
	Size       int64       `json:"size,omitempty"`
	ModTime    int64       `json:"mtime,omitempty"`
	ChangeTime int64       `json:"change_time,omitempty"`
}

// Older writers can add or replace stage files without receipts or a new
// publication token. Include unowned files to detect those changes.
func (c *checkpointCleanup) captureStage(ctx context.Context, stage *os.Root, files []File) {
	entries := map[string]checkpointStageEntry{}
	sidecars := map[string]checkpointStageSidecar{}
	payloads := make(map[string]bool, len(files))
	for _, file := range files {
		payloads[file.Path] = true
	}
	for _, transfer := range c.Transfers {
		if transfer.Path != "" {
			payloads[transfer.Path] = true
		}
	}
	digestPaths := make(map[string]bool, len(payloads))
	for path := range payloads {
		digestPaths[filepath.ToSlash(sidecarPath(filepath.FromSlash(path)))] = true
	}
	err := walkCheckpointStage(ctx, stage, func(name string, entry fs.DirEntry) error {
		state, err := readStageEntry(stage, name, entry)
		if err != nil {
			return err
		}
		entries[name] = state
		if state.Mode.IsRegular() && digestPaths[name] && !payloads[name] {
			if _, state, info := readStageSidecar(stage, filepath.FromSlash(name)); info != nil {
				sidecars[name] = state
			}
		}
		return nil
	})
	if err == nil {
		c.StageInventory, c.StageSidecars = entries, sidecars
	}
}

func (c checkpointCleanup) matchesStageInventory(ctx context.Context, stage *os.Root) bool {
	// Without a complete inventory there is no way to rule out an unrecorded legacy transfer.
	if len(c.StageInventory) == 0 {
		return false
	}
	err := walkCheckpointStage(ctx, stage, func(name string, entry fs.DirEntry) error {
		sealed, found := c.StageInventory[name]
		if !found {
			return fs.ErrInvalid
		}
		if _, sidecar := c.StageSidecars[name]; sidecar && entry.Type().IsRegular() {
			return nil // its independent identity and digest are checked before removal
		}
		current, err := readStageEntry(stage, name, entry)
		if err != nil {
			return err
		}
		if current != sealed || (current.Mode.IsRegular() && current.FileID == "") ||
			(!current.Mode.IsRegular() && !current.Mode.IsDir()) {
			return fs.ErrInvalid
		}
		return nil
	})
	return err == nil && ctx.Err() == nil
}

func readStageEntry(stage *os.Root, name string, entry fs.DirEntry) (checkpointStageEntry, error) {
	info, err := entry.Info()
	if err != nil {
		return checkpointStageEntry{}, err
	}
	state := checkpointStageEntry{Mode: info.Mode()}
	if !info.Mode().IsRegular() {
		return state, nil
	}
	file, err := openRegularDownloadFile(stage, filepath.FromSlash(name), os.O_RDONLY)
	if err != nil {
		return checkpointStageEntry{}, err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil {
		return checkpointStageEntry{}, err
	}
	if !sameCheckpointFileState(info, opened) || checkpointChangeTime(info) != checkpointChangeTime(opened) {
		return checkpointStageEntry{}, fs.ErrInvalid
	}
	state.FileID, state.Size = checkpointFileID(file, opened), opened.Size()
	state.ModTime, state.ChangeTime = opened.ModTime().UnixNano(), checkpointChangeTime(opened)
	return state, nil
}

func walkCheckpointStage(ctx context.Context, stage *os.Root, visit func(string, fs.DirEntry) error) error {
	return fs.WalkDir(stage.FS(), ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if name == "." || name == checkpointCleanupName {
			return nil
		}
		return visit(name, entry)
	})
}

// Only recognize the exact JSON format written by writeRootDigest. A sidecar
// with additional data is not ours to remove, even if its known fields match.
func readStageSidecar(stage *os.Root, path string) (digestSidecar, checkpointStageSidecar, os.FileInfo) {
	var record digestSidecar
	var state checkpointStageSidecar
	file, err := openRegularDownloadFile(stage, path, os.O_RDONLY)
	if err != nil {
		return record, state, nil
	}
	defer file.Close()
	const limit = 4096
	raw, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil || len(raw) > limit || json.Unmarshal(raw, &record) != nil || record.SHA256 == "" {
		return record, state, nil
	}
	canonical, err := json.Marshal(record)
	if err != nil || !bytes.Equal(raw, canonical) {
		return record, state, nil
	}
	info, err := file.Stat()
	if err != nil {
		return record, state, nil
	}
	id := checkpointFileID(file, info)
	if id == "" {
		return record, state, nil
	}
	return record, checkpointStageSidecar{FileID: id, ChangeTime: checkpointChangeTime(info), SHA256: checkpoint.ConfigDigest(raw)}, info
}

func (c checkpointCleanup) removeStageSidecar(stage *os.Root, name string, file os.FileInfo, digest string) {
	path := sidecarPath(name)
	sealed, ok := c.StageSidecars[filepath.ToSlash(path)]
	if !ok {
		return
	}
	record, state, info := readStageSidecar(stage, path)
	if info != nil && state == sealed && record.Size == file.Size() && record.ModTime == file.ModTime().UnixNano() && record.SHA256 == digest {
		removeUnchangedStageSidecar(stage, path, info)
	}
}

func removeUnchangedStageSidecar(stage *os.Root, path string, info os.FileInfo) {
	if current, err := stage.Lstat(path); err == nil && current.Mode().IsRegular() && sameCheckpointFileState(info, current) && checkpointChangeTime(info) == checkpointChangeTime(current) {
		_ = stage.Remove(path)
	}
}
