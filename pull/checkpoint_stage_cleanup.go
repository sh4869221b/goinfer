package pull

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

type checkpointStageSidecar struct {
	FileID     string `json:"file_id"`
	ChangeTime int64  `json:"change_time,omitempty"`
	SHA256     string `json:"sha256"`
}

// Older writers can add stage entries without receipts or a new publication token.
// Record even unowned entries so their presence can be distinguished from later work.
func (c *checkpointCleanup) captureStage(ctx context.Context, stage *os.Root) {
	var entries []string
	sidecars := map[string]checkpointStageSidecar{}
	err := walkCheckpointStage(ctx, stage, func(name string, entry fs.DirEntry) error {
		entries = append(entries, name)
		base := entry.Name()
		if entry.Type().IsRegular() && strings.HasPrefix(base, ".") && strings.HasSuffix(base, ".sha256") {
			if _, state, info := readStageSidecar(stage, filepath.FromSlash(name)); info != nil {
				sidecars[name] = state
			}
		}
		return nil
	})
	if err == nil {
		slices.Sort(entries)
		c.StageEntries, c.StageSidecars = entries, sidecars
	}
}

func (c checkpointCleanup) matchesStageEntries(ctx context.Context, stage *os.Root) bool {
	// Without a complete inventory there is no way to rule out an unrecorded legacy transfer.
	if len(c.StageEntries) == 0 {
		return false
	}
	contains := true
	err := walkCheckpointStage(ctx, stage, func(name string, _ fs.DirEntry) error {
		if _, found := slices.BinarySearch(c.StageEntries, name); !found {
			contains = false
			return fs.SkipAll
		}
		return nil
	})
	return contains && err == nil && ctx.Err() == nil
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
