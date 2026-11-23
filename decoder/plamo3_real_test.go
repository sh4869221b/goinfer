//go:build realckpt

package decoder

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

func TestPlamo3RealOracle(t *testing.T) {
	dir, golden := os.Getenv("PLAMO3_MODEL_DIR"), os.Getenv("PLAMO3_REAL_GOLDEN")
	if dir == "" || golden == "" {
		t.Skip("set PLAMO3_MODEL_DIR and PLAMO3_REAL_GOLDEN for the local real-checkpoint oracle")
	}
	raw, err := readGolden(golden)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		PromptIDs []int `json:"prompt_ids"`
		Logits    []float32
		Hidden    [][]float32
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("real-checkpoint reference has no cases")
	}
	m, err := Load(dir, Options{Backend: "cpu", Quant: "bf16"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	layers := make([]int, m.w.arch.NumLayers)
	for i := range layers {
		layers[i] = i
	}
	for ci, c := range cases {
		cache := m.NewCache(len(c.PromptIDs))
		var logits []float32
		var hidden [][]float32
		for _, id := range c.PromptIDs {
			logits, hidden, err = m.ForwardCapture(id, cache, layers)
			if err != nil {
				t.Fatal(err)
			}
		}
		if len(hidden) != len(layers) || len(c.Hidden) != len(layers) {
			t.Fatalf("case %d layer count mismatch: got %d, reference %d, want %d", ci, len(hidden), len(c.Hidden), len(layers))
		}
		for layer := range layers {
			cos, ok := plamo3OracleCosine(hidden[layer], c.Hidden[layer])
			if !ok {
				t.Fatalf("case %d layer %d cosine %g", ci, layer, cos)
			}
		}
		cos, ok := plamo3OracleCosine(logits, c.Logits)
		t.Logf("case %d: %d prompt tokens; %d layers; final logit cosine %.9f; argmax %d/%d", ci, len(c.PromptIDs), len(layers), cos, sampling.Argmax(logits), sampling.Argmax(c.Logits))
		if !ok || sampling.Argmax(logits) != sampling.Argmax(c.Logits) {
			t.Fatal("real-checkpoint logit parity failed")
		}
	}
}
