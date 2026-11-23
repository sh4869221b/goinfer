package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

func TestPlamo3Parity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/plamo3_tiny_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		PromptIDs    []int         `json:"prompt_ids"`
		Logits       [][]float32   `json:"logits"`
		Hidden       [][][]float32 `json:"hidden"`
		Continuation []int         `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load("../testdata/plamo3-tiny", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.w.arch
	if a.RMSAddOne || a.NormPlacement != NormSandwich4 || !a.QKNorm || a.isGlobalLayer(0) || !a.isGlobalLayer(1) {
		t.Fatalf("incorrect descriptor: %+v", a)
	}
	if a.ropeMscale(0) != 1 || math.Abs(a.ropeMscale(1)-(1+0.1*math.Log(256))) > 1e-9 {
		t.Fatal("incorrect per-layer YaRN scale")
	}
	cache := m.NewCache(len(g.PromptIDs) + len(g.Continuation))
	var logits []float32
	for pos, id := range g.PromptIDs {
		var hidden [][]float32
		logits, hidden, err = m.ForwardCapture(id, cache, []int{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if err := checkPlamo3TinyParity(logits, g.Logits[pos]); err != nil {
			t.Fatalf("position %d logits: %v", pos, err)
		}
		for layer := range hidden {
			if err := checkPlamo3TinyParity(hidden[layer], g.Hidden[layer][pos]); err != nil {
				t.Fatalf("position %d layer %d hidden state: %v", pos, layer, err)
			}
		}
	}
	for _, want := range g.Continuation {
		if got := sampling.Argmax(logits); got != want {
			t.Fatalf("greedy token %d, want %d", got, want)
		}
		logits, err = m.forward(want, cache)
		if err != nil {
			t.Fatal(err)
		}
	}
	out, errs := m.Generate(context.Background(), g.PromptIDs, len(g.Continuation), SamplingParams{})
	var got []int
	for id := range out {
		got = append(got, id)
	}
	if err := errs.Err(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, g.Continuation) {
		t.Fatalf("Generate %v, want %v", got, g.Continuation)
	}
	t.Log("all prompt positions, both layers, sliding-window wrap, YaRN and greedy generation match PFN fp32")
}

func checkPlamo3TinyParity(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("shape mismatch %d vs %d", len(got), len(want))
	}
	maxDiff := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			return fmt.Errorf("non-finite value at index %d: got %g, reference %g", i, got[i], want[i])
		}
		maxDiff = math.Max(maxDiff, diff)
	}
	if maxDiff > 2e-5 {
		return fmt.Errorf("maximum absolute difference %g exceeds 2e-5", maxDiff)
	}
	return nil
}

func TestPlamo3TinyParityRejectsNonFiniteValues(t *testing.T) {
	nan, posInf, negInf := float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))
	for _, tc := range []struct {
		name      string
		got, want []float32
	}{
		{"non-maximal logit NaN", []float32{3, nan, 1}, []float32{3, 2, 1}},
		{"reference NaN", []float32{3, 2, 1}, []float32{3, nan, 1}},
		{"positive infinity", []float32{3, posInf, 1}, []float32{3, 2, 1}},
		{"negative infinity", []float32{3, 2, 1}, []float32{3, negInf, 1}},
		{"matching infinities", []float32{3, posInf, 1}, []float32{3, posInf, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkPlamo3TinyParity(tc.got, tc.want); err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Errorf("comparison error = %v; want a non-finite value error", err)
			}
		})
	}
}

func TestPlamo3RejectsUnsupportedConfig(t *testing.T) {
	raw, err := os.ReadFile("../testdata/plamo3-tiny/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.PlamoLinearType = "fp8" },
		func(c *Config) { c.PlamoWindowSize = 0 },
		func(c *Config) { c.RopeParameters = nil },
		func(c *Config) { c.LayerTypes = []string{"mamba", "full_attention"} },
	} {
		var c Config
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatal(err)
		}
		mutate(&c)
		if _, _, err := resolveArchitecture(&c); err == nil {
			t.Fatal("unsupported config accepted")
		}
	}
}

func plamo3OracleCosine(got, want []float32) (float64, bool) {
	if len(got) != len(want) {
		return math.NaN(), false
	}
	cos := logitCosine(got, want)
	return cos, !math.IsInf(cos, 0) && cos >= 0.99999
}

func TestPlamo3OracleCosine(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want []float32
		pass      bool
	}{
		{"matching", []float32{1, 2}, []float32{2, 4}, true},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, false},
		{"zero actual", []float32{0, 0}, []float32{1, 2}, false},
		{"zero reference", []float32{1, 2}, []float32{0, 0}, false},
		{"both zero", []float32{0, 0}, []float32{0, 0}, false},
		{"non-finite", []float32{float32(math.Inf(1)), 1}, []float32{1, 1}, false},
		{"empty", nil, nil, false},
		{"different lengths", []float32{1, 2}, []float32{1}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if cos, pass := plamo3OracleCosine(tc.got, tc.want); pass != tc.pass {
				t.Errorf("cosine %g passed = %v; want %v", cos, pass, tc.pass)
			}
		})
	}
}
