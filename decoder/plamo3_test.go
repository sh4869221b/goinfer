package decoder

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
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
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s shape mismatch %d vs %d", label, len(got), len(want))
		}
		maxDiff := 0.0
		for i := range got {
			maxDiff = math.Max(maxDiff, math.Abs(float64(got[i]-want[i])))
		}
		if maxDiff > 2e-5 {
			t.Fatalf("%s maximum absolute difference %g exceeds 2e-5", label, maxDiff)
		}
	}
	for pos, id := range g.PromptIDs {
		var hidden [][]float32
		logits, hidden, err = m.ForwardCapture(id, cache, []int{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		check("logits", logits, g.Logits[pos])
		for layer := range hidden {
			check("hidden state", hidden[layer], g.Hidden[layer][pos])
		}
	}
	for _, want := range g.Continuation {
		if got := argmax(logits); got != want {
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
