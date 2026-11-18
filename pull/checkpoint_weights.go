package pull

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

const (
	checkpointWeights = "model.safetensors"
	checkpointIndex   = "model.safetensors.index.json"
	maxIndexBytes     = 64 << 20
)

// checkpointWeightPaths selects the same layout as the decoder, which prefers the index
// when present. Every shard named in the index must have a nonempty entry in files.
func checkpointWeightPaths(files map[string]File, readIndex func() ([]byte, error)) ([]string, error) {
	if _, ok := files[checkpointIndex]; !ok {
		if files[checkpointWeights].Size <= 0 {
			return nil, fmt.Errorf("checkpoint has no nonempty %s or %s", checkpointWeights, checkpointIndex)
		}
		return []string{checkpointWeights}, nil
	}
	raw, err := readIndex()
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", checkpointIndex, err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", checkpointIndex, err)
	}
	if len(index.WeightMap) == 0 {
		return nil, fmt.Errorf("%s has an empty weight_map", checkpointIndex)
	}
	seen := make(map[string]bool)
	for _, shard := range index.WeightMap {
		// The safetensors loader accepts shard filenames in the index's own directory.
		if !safeRepoPath(shard) || strings.Contains(shard, "/") || !filepath.IsLocal(shard) {
			return nil, fmt.Errorf("%s names unsafe shard path %q", checkpointIndex, shard)
		}
		if shard == checkpointIndex || shard == checkpointMarker || checkpointExtras[shard] {
			return nil, fmt.Errorf("%s names checkpoint metadata %q as a weight shard", checkpointIndex, shard)
		}
		if files[shard].Size <= 0 {
			return nil, fmt.Errorf("%s references shard %q without a nonempty checkpoint entry", checkpointIndex, shard)
		}
		seen[shard] = true
	}
	paths := make([]string, 0, len(seen)+1)
	for shard := range seen {
		paths = append(paths, shard)
	}
	slices.Sort(paths)
	return append(paths, checkpointIndex), nil
}

// verifyCheckpointWeights checks the on-disk layout against the manifest without hashing
// weights. Download and the full cache check verify file digests separately.
func verifyCheckpointWeights(root *os.Root, files []File, plannedIndex []byte) error {
	byPath := make(map[string]File, len(files))
	for _, f := range files {
		if !safeRepoPath(f.Path) || !filepath.IsLocal(filepath.FromSlash(f.Path)) {
			return fmt.Errorf("unsafe checkpoint path %q", f.Path)
		}
		if _, ok := byPath[f.Path]; ok {
			return fmt.Errorf("duplicate checkpoint entry %q", f.Path)
		}
		byPath[f.Path] = f
	}
	if _, ok := byPath["config.json"]; !ok {
		return fmt.Errorf("checkpoint has no config.json entry")
	}
	if _, indexed := byPath[checkpointIndex]; !indexed {
		// An unlisted index would take precedence over the single file in the decoder.
		if _, err := root.Stat(checkpointIndex); err == nil || !os.IsNotExist(err) {
			return fmt.Errorf("unexpected %s outside the checkpoint file list", checkpointIndex)
		}
	}
	paths, err := checkpointWeightPaths(byPath, func() ([]byte, error) {
		raw, err := checkpoint.ReadRootFile(root, checkpointIndex, maxIndexBytes)
		if err != nil {
			return nil, err
		}
		if plannedIndex != nil && !bytes.Equal(raw, plannedIndex) {
			return nil, fmt.Errorf("index differs from the current checkpoint plan")
		}
		index := byPath[checkpointIndex]
		if !checkpoint.ConfigMatches(raw, index.Size, index.SHA256) {
			return nil, fmt.Errorf("index does not match checkpoint metadata")
		}
		return raw, nil
	})
	if err != nil {
		return err
	}
	for _, name := range paths {
		if name == checkpointIndex {
			continue // checked the bytes read from the regular index file
		}
		info, err := checkpoint.StatRootFile(root, name)
		if err != nil {
			return fmt.Errorf("checking checkpoint weight %q: %w", name, err)
		}
		if !info.Mode().IsRegular() || info.Size() != byPath[name].Size {
			return fmt.Errorf("checkpoint weight %q is not a regular file of the recorded size", name)
		}
	}
	return nil
}
