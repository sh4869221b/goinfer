// Package checkpoint reads and verifies the config of a downloaded safetensors checkpoint.
package checkpoint

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	MarkerName     = ".goinfer-checkpoint.json"
	MaxConfigBytes = 4 << 20
)

// CacheRoot is the model directory shared by pull and on-disk model loaders.
func CacheRoot() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "goinfer", "models"), nil
}

func requiresMarker(dir string) bool {
	cache, err := CacheRoot()
	if err != nil {
		return false
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return false
	}
	if cacheCheckpointPath(cache, abs) {
		return true
	}
	cache, err = filepath.EvalSymlinks(cache)
	if err != nil {
		return false
	}
	abs, err = filepath.EvalSymlinks(abs)
	return err == nil && cacheCheckpointPath(cache, abs)
}

func cacheCheckpointPath(cache, dir string) bool {
	rel, err := filepath.Rel(cache, dir)
	return err == nil && filepath.IsLocal(rel) && strings.Count(filepath.ToSlash(rel), "/") == 1
}

// CheckLayout rejects layouts that the downloader cannot assemble.
func CheckLayout(repo, modelType string) error {
	if modelType == "plamo3" {
		return fmt.Errorf("repo %s uses PLaMo 3, which requires tokenizer.jsonl: built-in checkpoint pull does not support this layout; download the official checkpoint to a local directory and use demo/plamo-translate (docs/plamo3.md)", repo)
	}
	return nil
}

func ValidateConfig(repo string, raw []byte) error {
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("parsing config.json: %w", err)
	}
	return CheckLayout(repo, cfg.ModelType)
}

func ConfigMatches(raw []byte, size int64, digest string) bool {
	return raw != nil && int64(len(raw)) == size && (digest == "" || ConfigDigest(raw) == digest)
}

func ConfigDigest(raw []byte) string { return fmt.Sprintf("%x", sha256.Sum256(raw)) }

type Marker struct {
	Repo  string `json:"repo"`
	Files []struct {
		Path   string
		Size   int64
		SHA256 string
	} `json:"files"`
}

func ReadMarker(root *os.Root) (Marker, error) {
	var marker Marker
	f, err := root.OpenFile(MarkerName, readFlags, 0)
	if err != nil {
		return marker, fmt.Errorf("checkpoint marker: %w", err)
	}
	meta, err := readRegular(f)
	if err != nil {
		return marker, fmt.Errorf("checkpoint marker: %w", err)
	}
	if err := json.Unmarshal(meta, &marker); err != nil {
		return marker, fmt.Errorf("parsing checkpoint marker: %w", err)
	}
	return marker, nil
}

// ReadVerifiedConfig returns the exact bytes it verified against the marker. A local directory
// without a marker returns nil. Built-in cache paths require the marker even if it went missing.
func ReadVerifiedConfig(dir string) ([]byte, error) {
	raw, err := readVerifiedConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("cached checkpoint %s: %w", dir, err)
	}
	return raw, nil
}

func readVerifiedConfig(dir string) ([]byte, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	_, err = root.Lstat(MarkerName)
	switch {
	case os.IsNotExist(err) && !requiresMarker(dir):
		return nil, nil
	case err != nil:
		return nil, fmt.Errorf("checkpoint marker: %w", err)
	}
	marker, err := ReadMarker(root)
	if err != nil {
		return nil, err
	}
	raw, err := ReadRootConfig(root)
	if err != nil {
		return nil, fmt.Errorf("config.json must be a readable regular file inside the checkpoint directory: %w", err)
	}
	for _, file := range marker.Files {
		if file.Path == "config.json" {
			if !ConfigMatches(raw, file.Size, file.SHA256) {
				return nil, fmt.Errorf("config.json does not match the checkpoint marker")
			}
			if err := ValidateConfig(marker.Repo, raw); err != nil {
				return nil, err
			}
			return raw, nil
		}
	}
	return nil, fmt.Errorf("checkpoint marker has no config.json entry")
}

// ReadConfig verifies marked or built-in cached checkpoints. Other directories use the normal
// filesystem path, including local symlinks that point outside the directory.
func ReadConfig(dir string) ([]byte, error) {
	raw, err := ReadVerifiedConfig(dir)
	if raw != nil || err != nil {
		return raw, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "config.json"), readFlags, 0)
	if err != nil {
		return nil, err
	}
	return readRegular(f)
}

func readRegular(f *os.File) ([]byte, error) {
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", f.Name())
	}
	return io.ReadAll(f)
}
