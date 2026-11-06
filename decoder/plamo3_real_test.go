//go:build realckpt

package decoder

import (
	"encoding/json"
	"os"
	"testing"
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
		if len(c.Hidden) != len(hidden) || len(c.Logits) != len(logits) {
			t.Fatal("reference shape mismatch")
		}
		for layer := range layers {
			cos := logitCosine(hidden[layer], c.Hidden[layer])
			if cos < 0.99999 {
				t.Fatalf("case %d layer %d cosine %g", ci, layer, cos)
			}
		}
		cos := logitCosine(logits, c.Logits)
		t.Logf("case %d: %d prompt tokens; %d layers; final logit cosine %.9f; argmax %d/%d", ci, len(c.PromptIDs), len(layers), cos, argmax(logits), argmax(c.Logits))
		if cos < 0.99999 || argmax(logits) != argmax(c.Logits) {
			t.Fatal("real-checkpoint logit parity failed")
		}
	}
}
