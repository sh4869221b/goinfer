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
