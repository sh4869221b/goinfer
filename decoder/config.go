package decoder

import (
	"fmt"
	"io/fs"
	"os"

	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

// Config is a checkpoint's config.json, parsed. The type, its validators and its loader live in decoder/config, which
// imports no goinfer package; the name stays here as the published API.
type Config = config.Config

// RecommendedSampling is a checkpoint's own recommended sampling settings (config.RecommendedSampling).
type RecommendedSampling = config.RecommendedSampling

// loadConfig reads and parses config.json from fsys, and the recommended sampling beside it (config.Load).
func loadConfig(fsys fs.FS, name string) (*Config, error) { return config.Load(fsys, name) }

func loadDirConfig(dir string) (*Config, error) {
	b, err := checkpoint.ReadConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("decoder: read config.json: %w", err)
	}
	return config.Parse(os.DirFS(dir), "config.json", b)
}

// lenientConfigJSON makes a model config file's Python non-finite tokens parse (config.LenientJSON).
func lenientConfigJSON(b []byte) []byte { return config.LenientJSON(b) }
