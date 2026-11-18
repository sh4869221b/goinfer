package pull

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// fakeHF serves a repo's tree, its files (with Range, as HF's resolve host does) and its metadata, counts requests per
// file, and can fail one file's transfer midway.
type fakeHF struct {
	mu       sync.Mutex
	files    map[string][]byte
	gated    any
	hits     map[string]int
	ranges   map[string]string
	requests int
	failOnce string // a path whose next transfer is cut midway
}

func (h *fakeHF) serve(t *testing.T, repo string) {
	t.Helper()
	h.hits = map[string]int{}
	h.ranges = map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.requests++
		switch {
		case r.URL.Path == "/api/models/"+repo:
			_ = json.NewEncoder(w).Encode(map[string]any{"gated": h.gated, "private": false})
		case r.URL.Path == "/api/models/"+repo+"/tree/main":
			var out []map[string]any
			for p, b := range h.files {
				e := map[string]any{"type": "file", "path": p, "size": len(b)}
				if strings.HasSuffix(p, ".safetensors") || strings.HasSuffix(p, ".bin") || strings.HasSuffix(p, ".gguf") {
					s := sha256.Sum256(b)
					e["lfs"] = map[string]any{"oid": hex.EncodeToString(s[:])}
				}
				out = append(out, e)
			}
			_ = json.NewEncoder(w).Encode(out)
		case strings.HasPrefix(r.URL.Path, "/"+repo+"/resolve/main/"):
			p := strings.TrimPrefix(r.URL.Path, "/"+repo+"/resolve/main/")
			b, ok := h.files[p]
			if !ok {
				http.NotFound(w, r)
				return
			}
			h.hits[p]++
			h.ranges[p] = r.Header.Get("Range")
			if p == h.failOnce {
				h.failOnce = ""
				w.Header().Set("Content-Length", "999999999")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(b[:len(b)/2])
				return
			}
			http.ServeContent(w, r, p, time.Unix(0, 0), bytes.NewReader(b))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	oa, oc := hfAPI, hfCDN
	hfAPI, hfCDN = srv.URL+"/api/models", srv.URL
	t.Cleanup(func() { hfAPI, hfCDN = oa, oc })
}

// splitSafetensors splits one safetensors file into two (tensors alternated between them) and returns the shards and an
// index naming each tensor's shard, the shape of a real sharded checkpoint.
func splitSafetensors(t *testing.T, raw []byte) (a, b, index []byte) {
	t.Helper()
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	data := raw[8+n:]
	var names []string
	for k := range hdr {
		if k != "__metadata__" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	type ent struct {
		Dtype   string  `json:"dtype"`
		Shape   []int64 `json:"shape"`
		Offsets [2]int  `json:"data_offsets"`
	}
	build := func(keep func(i int) bool) []byte {
		h := map[string]ent{}
		var body []byte
		for i, k := range names {
			if !keep(i) {
				continue
			}
			var e ent
			if err := json.Unmarshal(hdr[k], &e); err != nil {
				t.Fatal(err)
			}
			chunk := data[e.Offsets[0]:e.Offsets[1]]
			e.Offsets = [2]int{len(body), len(body) + len(chunk)}
			body = append(body, chunk...)
			h[k] = e
		}
		hj, _ := json.Marshal(h)
		for len(hj)%8 != 0 {
			hj = append(hj, ' ')
		}
		out := make([]byte, 8, 8+len(hj)+len(body))
		binary.LittleEndian.PutUint64(out, uint64(len(hj)))
		return append(append(out, hj...), body...)
	}
	a, b = build(func(i int) bool { return i%2 == 0 }), build(func(i int) bool { return i%2 == 1 })
	wm := map[string]string{}
	for i, k := range names {
		wm[k] = map[bool]string{true: "model-00001-of-00002.safetensors", false: "model-00002-of-00002.safetensors"}[i%2 == 0]
	}
	index, _ = json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": wm})
	return a, b, index
}

// shardedTinyRepo is testdata/llama-tiny as a sharded HF repo, plus files a plan must leave out.
func shardedTinyRepo(t *testing.T) map[string][]byte {
	t.Helper()
	raw, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := os.ReadFile("../testdata/llama-tiny/config.json")
	if err != nil {
		t.Fatal(err)
	}
	a, b, idx := splitSafetensors(t, raw)
	return map[string][]byte{
		"config.json":                      cfg,
		"model.safetensors.index.json":     idx,
		"model-00001-of-00002.safetensors": a,
		"model-00002-of-00002.safetensors": b,
		"README.md":                        []byte("# a readme"),
		"pytorch_model.bin":                []byte("legacy weights a plan must not fetch"),
	}
}

func configWithDifferentEOS(t *testing.T, original []byte) []byte {
	t.Helper()
	updated := bytes.Replace(original, []byte(`"eos_token_id": 2`), []byte(`"eos_token_id": 3`), 1)
	if bytes.Equal(original, updated) || len(original) != len(updated) || !json.Valid(updated) {
		t.Fatal("fixture must have two different valid configs of equal size")
	}
	return updated
}

func withCacheRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", root)
	t.Setenv("HOME", root)         // os.UserCacheDir on darwin is $HOME/Library/Caches
	t.Setenv("LocalAppData", root) // and on Windows %LocalAppData%
	return root
}

func TestPlanCheckpointRejectsPlamo3BeforeWeights(t *testing.T) {
	for _, tc := range []struct {
		name      string
		tokenizer bool
	}{
		{name: "missing tokenizer"},
		{name: "with tokenizer", tokenizer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := map[string][]byte{
				"config.json":                      []byte(`{"model_type":"plamo3"}`),
				"generation_config.json":           []byte(`{}`),
				"tokenizer_config.json":            []byte(`{}`),
				"model.safetensors.index.json":     []byte(`{"weight_map":{"w":"model-00001-of-00001.safetensors"}}`),
				"model-00001-of-00001.safetensors": []byte("weights"),
			}
			if tc.tokenizer {
				files["tokenizer.jsonl"] = []byte(`["<|plamo:unk|>",0,"UNKNOWN"]`)
			}
			const repo = "o/plamo"
			h := &fakeHF{files: files}
			h.serve(t, repo)
			for _, plan := range []struct {
				name string
				run  func(context.Context, string) (Plan, error)
			}{
				{"generative", PlanCheckpoint},
				{"pull CLI", func(ctx context.Context, repo string) (Plan, error) {
					return PlanCheckpointFor(ctx, repo, AnyLoads)
				}},
			} {
				p, err := plan.run(t.Context(), repo)
				if err == nil || !strings.Contains(err.Error(), "tokenizer.jsonl") || !strings.Contains(err.Error(), "local directory") {
					t.Errorf("%s plan = %+v, %v; want a local-directory error", plan.name, p, err)
				}
				if len(p.Files) != 0 {
					t.Errorf("%s planned files despite rejection: %v", plan.name, p.Files)
				}
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if len(h.hits) != 1 || h.hits["config.json"] != 2 {
				t.Errorf("requested files = %v; only config.json should be fetched", h.hits)
			}
		})
	}
}

func TestResolveLegacyCheckpointCache(t *testing.T) {
	for _, tc := range []struct {
		name, modelType string
		link, wantErr   string
	}{
		{name: "PLaMo regular", modelType: "plamo3", wantErr: "tokenizer.jsonl"},
		{name: "Llama regular", modelType: "llama"},
		{name: "PLaMo relative link", modelType: "plamo3", link: "relative", wantErr: "tokenizer.jsonl"},
		{name: "Llama relative link", modelType: "llama", link: "relative"},
		{name: "PLaMo absolute local link", modelType: "plamo3", link: "absolute", wantErr: "tokenizer.jsonl"},
		{name: "Llama absolute local link", modelType: "llama", link: "absolute"},
		{name: "Llama chained absolute local link", modelType: "llama", link: "chained absolute"},
		{name: "PLaMo chained absolute local link", modelType: "plamo3", link: "chained absolute", wantErr: "tokenizer.jsonl"},
		{name: "Llama cache root alias", modelType: "llama", link: "cache alias"},
		{name: "PLaMo cache root alias", modelType: "plamo3", link: "cache alias", wantErr: "tokenizer.jsonl"},
		{name: "Llama absolute directory link", modelType: "llama", link: "absolute directory"},
		{name: "Llama absolute link to checkpoint directory", modelType: "llama", link: "absolute checkpoint directory"},
		{name: "broken link", modelType: "llama", link: "broken", wantErr: "config.json"},
		{name: "relative escape", modelType: "plamo3", link: "relative escape", wantErr: "config.json"},
		{name: "absolute escape", modelType: "plamo3", link: "absolute escape", wantErr: "config.json"},
		{name: "chained escape", modelType: "plamo3", link: "chained escape", wantErr: "config.json"},
		{name: "escape and reentry", modelType: "plamo3", link: "reentry", wantErr: "config.json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withCacheRoot(t)
			const repo = "o/legacy"
			dir, err := CacheDir(repo)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			config := []byte(fmt.Sprintf(`{"model_type":%q}`, tc.modelType))
			weights := []byte("legacy weights")
			marker, err := json.Marshal(markerDoc{Repo: repo, Files: []File{
				{Path: "config.json", Size: int64(len(config))},
				{Path: "model.safetensors", Size: int64(len(weights)), SHA256: fmt.Sprintf("%x", sha256.Sum256(weights))},
			}})
			if err != nil {
				t.Fatal(err)
			}
			configPath := filepath.Join(dir, "config.json")
			configFile := configPath
			var linkTarget string
			switch tc.link {
			case "relative", "absolute", "chained absolute":
				configFile = filepath.Join(dir, "model-config.json")
				linkTarget = filepath.Base(configFile)
				if tc.link != "relative" {
					linkTarget, err = filepath.EvalSymlinks(dir)
					if err != nil {
						t.Fatal(err)
					}
					linkTarget = filepath.Join(linkTarget, filepath.Base(configFile))
				}
				if tc.link == "chained absolute" {
					saveCheckpointConfigLink(t, linkTarget, filepath.Join(dir, "active.json"))
					linkTarget = "active.json"
				}
			case "cache alias":
				cache, err := CacheRoot()
				if err != nil {
					t.Fatal(err)
				}
				alias := filepath.Join(t.TempDir(), "models-alias")
				saveCheckpointConfigLink(t, cache, alias)
				configFile = filepath.Join(dir, "model-config.json")
				linkTarget = filepath.Join(alias, filepath.FromSlash(repo), filepath.Base(configFile))
			case "absolute directory", "absolute checkpoint directory":
				dataDir := dir
				if tc.link == "absolute directory" {
					dataDir = filepath.Join(dir, "data")
					if err := os.Mkdir(dataDir, 0o755); err != nil {
						t.Fatal(err)
					}
				}
				saveCheckpointConfigLink(t, dataDir, filepath.Join(dir, "active"))
				configFile = filepath.Join(dataDir, "model-config.json")
				linkTarget = filepath.Join("active", filepath.Base(configFile))
			case "reentry":
				outside := t.TempDir()
				saveCheckpointConfigLink(t, outside, filepath.Join(dir, "outside"))
				saveCheckpointConfigLink(t, dir, filepath.Join(outside, "reentry"))
				configFile = filepath.Join(dir, "model-config.json")
				linkTarget = filepath.Join(dir, "outside", "reentry", filepath.Base(configFile))
			case "broken":
				configFile, linkTarget = "", "missing.json"
			case "relative escape", "absolute escape", "chained escape":
				configFile = filepath.Join(t.TempDir(), "outside-config.json")
				linkTarget = configFile
				if tc.link == "relative escape" {
					linkTarget, err = filepath.Rel(dir, configFile)
					if err != nil {
						t.Fatal(err)
					}
				} else if tc.link == "chained escape" {
					saveCheckpointConfigLink(t, configFile, filepath.Join(dir, "intermediate.json"))
					linkTarget = "intermediate.json"
				}
			}
			files := map[string][]byte{filepath.Join(dir, "model.safetensors"): weights, filepath.Join(dir, checkpointMarker): marker}
			if configFile != "" {
				files[configFile] = config
			}
			for name, data := range files {
				if err := os.WriteFile(name, data, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if linkTarget != "" {
				saveCheckpointConfigLink(t, linkTarget, configPath)
			}
			h := &fakeHF{}
			h.serve(t, repo)
			resolved, err := Resolve(t.Context(), "hf:"+repo+":safetensors", nil)
			cached, ok := CachedCheckpoint(dir)
			if tc.wantErr != "" {
				if resolved != "" || err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Resolve = %q, %v; want refusal mentioning %q", resolved, err, tc.wantErr)
				} else if tc.wantErr == "tokenizer.jsonl" && !strings.Contains(err.Error(), "local directory") {
					t.Errorf("Resolve error lacks local-directory guidance: %v", err)
				} else if tc.wantErr == "config.json" && strings.Contains(err.Error(), "tokenizer.jsonl") {
					t.Errorf("Resolve diagnosed PLaMo through an unusable config link: %v", err)
				}
				if cached != "" || ok {
					t.Errorf("CachedCheckpoint = %q, %v; want refusal", cached, ok)
				}
			} else if err != nil || resolved != dir || !ok || cached != dir {
				t.Errorf("Resolve = %q, %v and CachedCheckpoint = %q, %v; want %q", resolved, err, cached, ok, dir)
			}
			h.mu.Lock()
			requests := h.requests
			h.mu.Unlock()
			if requests != 0 {
				t.Errorf("cache lookup made %d network requests", requests)
			}
			for name, want := range files {
				got, err := os.ReadFile(name)
				if err != nil || !bytes.Equal(got, want) {
					t.Errorf("cache file %s changed: %v", name, err)
				}
			}
			if linkTarget != "" {
				if got, err := os.Readlink(configPath); err != nil || got != linkTarget {
					t.Errorf("config link changed: got %q, %v; want %q", got, err, linkTarget)
				}
			}
		})
	}
}

func TestReadCachedCheckpointConfigWhileAliasChanges(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires renaming over an existing symlink")
	}
	dir, outside := t.TempDir(), t.TempDir()
	const insideConfig = `{"model_type":"llama"}`
	const outsideConfig = `{"model_type":"plamo3"}`
	for dest, config := range map[string]string{dir: insideConfig, outside: outsideConfig} {
		if err := os.WriteFile(filepath.Join(dest, "model-config.json"), []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	aliasDir := t.TempDir()
	alias := filepath.Join(aliasDir, "checkpoint-alias")
	saveCheckpointConfigLink(t, dir, alias)
	saveCheckpointConfigLink(t, filepath.Join(alias, "model-config.json"), filepath.Join(dir, "active.json"))
	saveCheckpointConfigLink(t, "active.json", filepath.Join(dir, "config.json"))
	if got, err := readCheckpointMetadata(dir, "config.json"); err != nil || string(got) != insideConfig {
		t.Fatalf("initial config = %q, %v; want in-checkpoint data", got, err)
	}

	const attempts = 200
	done := make(chan error, 1)
	go func() {
		next := filepath.Join(aliasDir, "next")
		for i := range attempts {
			target := outside
			if i%2 == 1 {
				target = dir
			}
			if err := os.Symlink(target, next); err != nil {
				done <- err
				return
			}
			if err := os.Rename(next, alias); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	for range attempts {
		got, err := readCheckpointMetadata(dir, "config.json")
		if err == nil && string(got) != insideConfig {
			t.Errorf("read config outside checkpoint while its alias changed: %q", got)
			break
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func saveCheckpointConfigLink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
}

func TestPublicCheckpointLoadsRecheckConfig(t *testing.T) {
	withCacheRoot(t)
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	dir, err := Resolve(t.Context(), "hf:"+repo+":safetensors", nil)
	if err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(dir, "config.json")
	outside := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(outside, files["config.json"], 0o644); err != nil {
		t.Fatal(err)
	}
	loaders := []struct {
		name string
		load func(string) error
	}{
		{"Load", func(p string) error {
			m, err := decoder.Load(p, decoder.Options{DirectLoad: true})
			if err != nil {
				return err
			}
			return m.Close()
		}},
		{"LoadWeights", func(p string) error {
			_, err := decoder.LoadWeights(p)
			return err
		}},
		{"StreamTranscodeDir", func(p string) error {
			_, err := decoder.StreamTranscodeDir(t.Context(), p, io.Discard, "f32", false, decoder.GIWTargetNone, "tiny")
			return err
		}},
	}
	checkRefusal := func(t *testing.T, dir, want string) {
		t.Helper()
		for _, loader := range loaders {
			t.Run(loader.name, func(t *testing.T) {
				if err := loader.load(dir); err == nil || !strings.Contains(err.Error(), want) {
					t.Fatalf("want error containing %q, got %v", want, err)
				}
			})
		}
	}
	t.Run("outside link after Resolve", func(t *testing.T) {
		if err := os.Remove(configPath); err != nil {
			t.Fatal(err)
		}
		saveCheckpointConfigLink(t, outside, configPath)
		checkRefusal(t, dir, "inside the checkpoint directory")
		if err := os.Remove(configPath); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("same-size config change", func(t *testing.T) {
		original := files["config.json"]
		changed := bytes.Replace(original, []byte(`"eos_token_id": 2`), []byte(`"eos_token_id": 3`), 1)
		if bytes.Equal(changed, original) || len(changed) != len(original) {
			t.Fatal("fixture must change content without changing length")
		}
		if err := os.WriteFile(configPath, changed, 0o644); err != nil {
			t.Fatal(err)
		}
		checkRefusal(t, dir, "config.json does not match the checkpoint marker")
		if err := os.WriteFile(configPath, original, 0o644); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("missing marker after Resolve", func(t *testing.T) {
		if err := os.Remove(filepath.Join(dir, checkpointMarker)); err != nil {
			t.Fatal(err)
		}
		checkRefusal(t, dir, "checkpoint marker")
		alias := filepath.Join(t.TempDir(), "alias")
		saveCheckpointConfigLink(t, dir, alias)
		checkRefusal(t, alias, "checkpoint marker")
	})
	t.Run("local directory keeps external config links", func(t *testing.T) {
		local := filepath.Join(t.TempDir(), "local")
		if err := os.Rename(dir, local); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(local, "config.json")); err != nil {
			t.Fatal(err)
		}
		saveCheckpointConfigLink(t, outside, filepath.Join(local, "config.json"))
		if err := loaders[0].load(local); err != nil {
			t.Fatal(err)
		}
	})
}

// TestCheckpoint_shardedRoundTrip is P9's first gate (docs/tasks/task-checkpoint-fetch-2026-09.md): a safetensors
// repo with a shard index goes plan -> fetch -> load -> one token. The plan takes config, index and both shards and
// leaves the README and the legacy .bin out; the fetched directory is what decoder.Load opens; a second Resolve is
// answered from the cache with no request.
func TestCheckpoint_shardedRoundTrip(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false}
	h.serve(t, "o/tiny")
	p, err := PlanCheckpoint(context.Background(), "o/tiny")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	var paths []string
	for _, f := range p.Files {
		paths = append(paths, f.Path)
	}
	want := "config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json"
	if strings.Join(paths, " ") != want || p.Family != "llama" {
		t.Fatalf("plan files %v family %q; want %s and llama", paths, p.Family, want)
	}
	dir, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if _, err := os.Stat(dir + ".partial"); !os.IsNotExist(err) {
		t.Fatalf("the staging directory survived a completed pull: %v", err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("decoder.Load on the fetched directory: %v", err)
	}
	defer m.Close()
	ch, g := m.Generate(context.Background(), []int{1, 2, 3}, 1, decoder.SamplingParams{})
	n := 0
	for range ch {
		n++
	}
	if g.Err() != nil || n != 1 {
		t.Fatalf("one token from the fetched checkpoint: %d tokens, err %v", n, g.Err())
	}
	before := 0
	for _, c := range h.hits {
		before += c
	}
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err != nil {
		t.Fatalf("second resolve: %v", err)
	}
	after := 0
	for _, c := range h.hits {
		after += c
	}
	if after != before {
		t.Fatalf("a second resolve of a complete checkpoint fetched %d more files", after-before)
	}
}

// TestCheckpoint_interruptedSetNeverPublishes is the gate the doc names as mattering most: a transfer that dies inside
// the set leaves NOTHING at the final path (only the staging directory), and the re-run resumes without re-fetching
// the files already verified.
func TestCheckpoint_interruptedSetNeverPublishes(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false, failOnce: "model-00002-of-00002.safetensors"}
	h.serve(t, "o/tiny")
	dir, _ := CacheDir("o/tiny")
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil {
		t.Fatal("a transfer cut midway reported success")
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("an incomplete set published %s (stat err %v)", dir, err)
	}
	if _, err := os.Stat(dir + ".partial"); err != nil {
		t.Fatalf("the staging directory is gone, so nothing can resume: %v", err)
	}
	firstShard := h.hits["model-00001-of-00002.safetensors"]
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if h.hits["model-00001-of-00002.safetensors"] != firstShard {
		t.Fatalf("the resume re-fetched shard 1 (%d -> %d requests)", firstShard, h.hits["model-00001-of-00002.safetensors"])
	}
	if _, ok := CachedCheckpoint(dir); !ok {
		t.Fatal("the resumed set did not publish a complete checkpoint")
	}
}

func TestCheckpointResumeAfterWeightLayoutChange(t *testing.T) {
	const repo = "o/tiny"
	const tokenizerFile = "tokenizer_config.json"
	const firstShard = "model-00001-of-00002.safetensors"
	const secondShard = "model-00002-of-00002.safetensors"
	for _, tc := range []struct {
		name                               string
		toSingle, linkedIndex, reuseSingle bool
	}{
		{name: "split to single", toSingle: true},
		{name: "split to single with linked index", toSingle: true, linkedIndex: true},
		{name: "single to split"},
		{name: "single to split that reuses the single file", reuseSingle: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withCacheRoot(t)
			split := shardedTinyRepo(t)
			split[tokenizerFile] = []byte(`{"model_max_length":2048,"padding_side":"right"}`)
			weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			single := map[string][]byte{"config.json": split["config.json"], tokenizerFile: split[tokenizerFile], checkpointWeights: weights}
			if tc.reuseSingle {
				split[checkpointIndex] = bytes.ReplaceAll(split[checkpointIndex], []byte(firstShard), []byte(checkpointWeights))
				delete(split, firstShard)
				split[checkpointWeights] = weights
			}
			old, current := single, split
			if tc.toSingle {
				old, current = split, single
			}
			h := &fakeHF{files: old, failOnce: tokenizerFile}
			h.serve(t, repo)
			ctx := t.Context()
			dest, err := CacheDir(repo)
			if err != nil {
				t.Fatal(err)
			}
			oldPlan, download, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := download(ctx, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), tokenizerFile) {
				t.Fatalf("expected interruption during %s, got %q, %v", tokenizerFile, got, err)
			}
			staging := dest + ".partial"
			part, err := os.Stat(filepath.Join(staging, tokenizerFile+".part"))
			if err != nil {
				t.Fatal(err)
			}
			if tc.toSingle {
				if raw, err := os.ReadFile(filepath.Join(staging, checkpointIndex)); err != nil || !bytes.Equal(raw, old[checkpointIndex]) {
					t.Fatalf("interrupted pull did not stage the old index: %v", err)
				}
			}
			if tc.linkedIndex {
				stagedIndex := filepath.Join(staging, checkpointIndex)
				outside := filepath.Join(t.TempDir(), checkpointIndex)
				if err := os.Rename(stagedIndex, outside); err != nil {
					t.Fatal(err)
				}
				saveCheckpointConfigLink(t, outside, stagedIndex)
				t.Cleanup(func() {
					if raw, err := os.ReadFile(outside); err != nil || !bytes.Equal(raw, old[checkpointIndex]) {
						t.Errorf("cleanup modified the external index target: %v", err)
					}
				})
			}
			h.mu.Lock()
			h.files = current
			h.mu.Unlock()
			plan, err := PlanCheckpoint(ctx, repo)
			if err != nil {
				t.Fatal(err)
			}
			if err := plan.VerifyPin(oldPlan.TreeDigest()); err == nil {
				t.Fatal("a new weight layout matched the previous tree pin")
			}
			if got, err := Resolve(ctx, "hf:"+repo+":safetensors", nil); got != dest || err != nil {
				t.Fatalf("resume after layout change = %q, %v; want %s", got, err, dest)
			}
			if _, ok := CachedCheckpoint(dest); !ok {
				t.Fatal("resumed checkpoint failed the offline cache check")
			}
			model, err := decoder.Load(dest, decoder.Options{Backend: "cpu"})
			if err != nil {
				t.Fatal(err)
			}
			model.Close()
			marker := readTestCheckpointMarker(t, dest)
			if tc.toSingle {
				if _, err := os.Lstat(filepath.Join(dest, checkpointIndex)); !os.IsNotExist(err) {
					t.Errorf("obsolete index survived publication: %v", err)
				}
				for _, shard := range []string{firstShard, secondShard} {
					if _, err := os.Stat(filepath.Join(dest, shard)); err != nil {
						t.Errorf("cleanup removed previously downloaded shard %s: %v", shard, err)
					}
				}
			}
			if listed := slices.ContainsFunc(marker.Files, func(f File) bool { return f.Path == checkpointWeights }); listed != (tc.toSingle || tc.reuseSingle) {
				t.Errorf("single file manifest entry = %t", listed)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if got := h.ranges[tokenizerFile]; got != fmt.Sprintf("bytes=%d-", part.Size()) {
				t.Errorf("tokenizer resume Range = %q", got)
			}
			if h.hits[checkpointWeights] != 1 {
				t.Errorf("single weight file should be downloaded exactly once across both pulls: %v", h.hits)
			}
		})
	}
}

func TestCheckpointLayoutCleanupLeavesOtherDirectoriesAlone(t *testing.T) {
	const repo = "o/tiny"
	const spec = "hf:o/tiny:safetensors"
	for _, completed := range []bool{false, true} {
		t.Run(fmt.Sprintf("completed=%t", completed), func(t *testing.T) {
			withCacheRoot(t)
			split := shardedTinyRepo(t)
			weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			h := &fakeHF{files: map[string][]byte{"config.json": split["config.json"], checkpointWeights: weights}}
			h.serve(t, repo)
			ctx := t.Context()
			dest, err := CacheDir(repo)
			if err != nil {
				t.Fatal(err)
			}
			_, download, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			var indexPath, wantError string
			if completed {
				if _, err := download(ctx, dest, nil); err != nil {
					t.Fatal(err)
				}
				indexPath = filepath.Join(dest, checkpointIndex)
				wantError = "unexpected " + checkpointIndex
			} else {
				if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
					t.Fatal(err)
				}
				outside := t.TempDir()
				saveCheckpointConfigLink(t, outside, dest+".partial")
				indexPath = filepath.Join(outside, checkpointIndex)
				wantError = "opening staging directory"
			}
			if err := os.WriteFile(indexPath, split[checkpointIndex], 0o644); err != nil {
				t.Fatal(err)
			}
			h.mu.Lock()
			requests := h.requests
			h.mu.Unlock()
			if path, err := download(ctx, dest, nil); path != "" || err == nil || !strings.Contains(err.Error(), wantError) {
				t.Errorf("download = %q, %v; want refusal mentioning %q", path, err, wantError)
			}
			if raw, err := os.ReadFile(indexPath); err != nil || !bytes.Equal(raw, split[checkpointIndex]) {
				t.Errorf("refusal modified the index: %v", err)
			}
			if completed {
				if _, ok := CachedCheckpoint(dest); ok {
					t.Error("CachedCheckpoint accepted a completed directory with an unlisted index")
				}
				if path, err := Resolve(ctx, spec, nil); path != "" || err == nil || !strings.Contains(err.Error(), wantError) {
					t.Errorf("Resolve = %q, %v; want the same refusal", path, err)
				}
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.requests != requests {
				t.Errorf("refusing unsafe staging or completed cache made %d requests", h.requests-requests)
			}
		})
	}
}

func TestCheckpointLayoutCleanupRejectsReplacedStaging(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires renaming an open staging directory")
	}
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	const shard = "model-00001-of-00002.safetensors"
	h := &fakeHF{files: map[string][]byte{"config.json": files["config.json"], checkpointWeights: files[shard]}}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "checkpoint")
	staging := dest + ".partial"
	if err := os.Mkdir(staging, 0o755); err != nil {
		t.Fatal(err)
	}
	outside, moved := t.TempDir(), staging+".moved"
	for _, dir := range []string{staging, outside} {
		if err := os.WriteFile(filepath.Join(dir, checkpointIndex), files[checkpointIndex], 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var replaced bool
	progress := func(done, total int64, name string) {
		if replaced || done != total || name != checkpointWeights {
			return
		}
		if _, err := os.Stat(filepath.Join(staging, checkpointWeights)); err != nil {
			return // wait for the callback after Download renames the file
		}
		if err := os.Rename(staging, moved); err != nil {
			t.Fatal(err)
		}
		saveCheckpointConfigLink(t, outside, staging)
		replaced = true
	}
	if path, err := download(t.Context(), dest, progress); !replaced || path != "" || err == nil || !strings.Contains(err.Error(), "staging directory changed") {
		t.Fatalf("download = %q, %v after staging replacement=%t; want refusal", path, err, replaced)
	}
	if _, err := os.Lstat(dest); !os.IsNotExist(err) {
		t.Errorf("the replacement staging path was published: %v", err)
	}
	for _, dir := range []string{moved, outside} {
		if raw, err := os.ReadFile(filepath.Join(dir, checkpointIndex)); err != nil || !bytes.Equal(raw, files[checkpointIndex]) {
			t.Errorf("refusal modified the index in %s: %v", dir, err)
		}
	}
}

func TestCheckpointResumeUsesPlannedConfig(t *testing.T) {
	const repo = "o/tiny"
	const firstShard = "model-00001-of-00002.safetensors"
	const resumedShard = "model-00002-of-00002.safetensors"
	for _, malformed := range []bool{false, true} {
		name := "different valid JSON"
		if malformed {
			name = "malformed JSON"
		}
		t.Run(name, func(t *testing.T) {
			files := shardedTinyRepo(t)
			original := files["config.json"]
			updated := configWithDifferentEOS(t, original)
			h := &fakeHF{files: files, failOnce: resumedShard}
			h.serve(t, repo)
			ctx := t.Context()
			dest := filepath.Join(t.TempDir(), "checkpoint")
			oldPlan, oldDownload, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := oldDownload(ctx, dest, nil); err == nil {
				t.Fatal("first download should stop on the interrupted shard")
			}
			staging := dest + ".partial"
			part, err := os.Stat(filepath.Join(staging, resumedShard+".part"))
			if err != nil || part.Size() <= 0 {
				t.Fatalf("expected a partially downloaded shard, stat=%v (%v)", part, err)
			}
			if malformed {
				bad := bytes.Clone(original)
				bad[0] = '!'
				if err := os.WriteFile(filepath.Join(staging, "config.json"), bad, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			h.mu.Lock()
			h.files["config.json"] = updated
			firstHits := h.hits[firstShard]
			h.mu.Unlock()
			plan, download, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			pin := oldPlan.TreeDigest()
			if err := plan.VerifyPin(pin); err != nil {
				t.Fatalf("same-size non-LFS config changed the tree pin: %v", err)
			}
			plan.Files[0].Path = "metadata-only.json"
			h.mu.Lock()
			configHits := h.hits["config.json"]
			h.files["config.json"] = original // the plan still owns the bytes it validated
			h.mu.Unlock()
			if _, err := download(ctx, dest, nil); err != nil {
				t.Fatalf("resume: %v", err)
			}
			got, err := os.ReadFile(filepath.Join(dest, "config.json"))
			if err != nil || !bytes.Equal(got, updated) {
				t.Fatalf("published config differs from the planned bytes (err %v)", err)
			}
			root, err := os.OpenRoot(dest)
			if err != nil {
				t.Fatal(err)
			}
			defer root.Close()
			marker, ok := readCheckpointMarker(root)
			if !ok || marker.Files[0].Path != "config.json" || marker.Files[0].SHA256 != fmt.Sprintf("%x", sha256.Sum256(updated)) {
				t.Fatalf("marker did not record the planned config's SHA-256: %+v", marker)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.hits["config.json"] != configHits || h.hits[firstShard] != firstHits {
				t.Errorf("re-fetched the planned config or completed shard: %v", h.hits)
			}
			if got, want := h.ranges[resumedShard], fmt.Sprintf("bytes=%d-", part.Size()); got != want {
				t.Errorf("resumed shard request range = %q, want %q", got, want)
			}
		})
	}
}

func TestDownloadCheckpointRejectsCompletedConfigDrift(t *testing.T) {
	const repo = "o/tiny"
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	original := files["config.json"]
	h := &fakeHF{files: files}
	h.serve(t, repo)
	ctx := t.Context()
	dest, err := CacheDir(repo)
	if err != nil {
		t.Fatal(err)
	}
	oldPlan, oldDownload, err := PrepareCheckpointFor(ctx, repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := oldDownload(ctx, dest, nil); err != nil {
		t.Fatal(err)
	}
	marker, err := os.ReadFile(filepath.Join(dest, checkpointMarker))
	if err != nil {
		t.Fatal(err)
	}
	updated := configWithDifferentEOS(t, original)
	h.mu.Lock()
	h.files["config.json"] = updated
	h.mu.Unlock()
	plan, download, err := PrepareCheckpointFor(ctx, repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.VerifyPin(oldPlan.TreeDigest()); err != nil {
		t.Fatalf("same-size config changed the tree pin: %v", err)
	}
	h.mu.Lock()
	requests := h.requests
	h.mu.Unlock()
	if got, err := oldDownload(ctx, dest, nil); err != nil || got != dest {
		t.Errorf("matching plan should reuse the completed checkpoint: %q, %v", got, err)
	}
	if got, err := download(ctx, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "differs from the current checkpoint plan") {
		t.Errorf("changed config should refuse the completed destination: %q, %v", got, err)
	}
	if got, err := DownloadCheckpoint(ctx, oldPlan, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "differs from the current checkpoint plan") {
		t.Errorf("download from metadata should also refuse the completed destination: %q, %v", got, err)
	}
	if got, err := Resolve(ctx, "hf:"+repo+":safetensors", nil); err != nil || got != dest {
		t.Errorf("offline Resolve should still open the verified cache: %q, %v", got, err)
	}
	if _, err := os.Stat(dest + ".partial"); !os.IsNotExist(err) {
		t.Errorf("refusal left a staging directory: %v", err)
	}
	for _, f := range oldPlan.Files {
		want := files[f.Path]
		if f.Path == "config.json" {
			want = original
		}
		if got, err := os.ReadFile(filepath.Join(dest, f.Path)); err != nil || !bytes.Equal(got, want) {
			t.Errorf("refusal changed %s: %v", f.Path, err)
		}
	}
	if got, err := os.ReadFile(filepath.Join(dest, checkpointMarker)); err != nil || !bytes.Equal(got, marker) {
		t.Errorf("refusal changed the marker: %v", err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.requests != requests+2 {
		t.Errorf("expected only the config and index revalidation requests, got %d requests", h.requests-requests)
	}
}

func TestPlanJSONEmbedding(t *testing.T) {
	type request struct {
		Plan
		Destination string
	}
	want := request{
		Plan:        Plan{Repo: "o/tiny", Files: []File{{Path: "config.json", Size: 2}}, Bytes: 2, ModelType: "llama", Family: "llama", Dtype: "float32"},
		Destination: "models/tiny",
	}
	const wire = `{"Repo":"o/tiny","Files":[{"Path":"config.json","Size":2,"SHA256":""}],"Bytes":2,"ModelType":"llama","Family":"llama","Dtype":"float32","Destination":"models/tiny"}`
	if got, err := json.Marshal(want); err != nil || string(got) != wire {
		t.Fatalf("embedded Plan JSON = %s, %v; want %s", got, err, wire)
	}
	var got request
	if err := json.Unmarshal([]byte(wire), &got); err != nil {
		t.Fatal(err)
	}
	if encoded, err := json.Marshal(got); err != nil || string(encoded) != wire {
		t.Fatalf("round-trip JSON = %s, %v; want %s", encoded, err, wire)
	}
}

func TestPlanFilesHasOriginalSliceType(t *testing.T) {
	if _, ok := any(Plan{}.Files).([]File); !ok {
		t.Fatalf("Plan.Files has type %T; want []File", Plan{}.Files)
	}
	type callerFiles []File
	plan := Plan{Files: callerFiles{{Path: "config.json"}}}
	var files callerFiles = plan.Files
	if len(files) != 1 || files[0].Path != "config.json" {
		t.Fatalf("assignment through named slice = %v", files)
	}
}

func TestDownloadCheckpointConfigAfterPartialJSON(t *testing.T) {
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	original := files["config.json"]
	updated := configWithDifferentEOS(t, original)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	plan, err := PlanCheckpoint(t.Context(), repo)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"Family":"llama"}`), &plan); err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.files["config.json"] = updated
	h.mu.Unlock()
	dest := filepath.Join(t.TempDir(), "checkpoint")
	if _, err := DownloadCheckpoint(t.Context(), plan, dest, nil); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dest, "config.json"))
	if err != nil || !bytes.Equal(got, updated) {
		t.Fatalf("download did not use current config after decoding JSON without Files (read error %v)", err)
	}
}

func TestDownloadCheckpointClonedFiles(t *testing.T) {
	const repo = "o/tiny"
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, repo)
	plan, err := PlanCheckpoint(t.Context(), repo)
	if err != nil {
		t.Fatal(err)
	}
	plan.Files = slices.Clone(plan.Files)
	if _, err := DownloadCheckpoint(t.Context(), plan, filepath.Join(t.TempDir(), "checkpoint"), nil); err != nil {
		t.Fatalf("download from a cloned file list: %v", err)
	}
}

func TestDownloadCheckpointRequiresConfigEntry(t *testing.T) {
	const repo = "o/tiny"
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, repo)
	plan, err := PlanCheckpoint(t.Context(), repo)
	if err != nil {
		t.Fatal(err)
	}
	plan.Files = slices.DeleteFunc(plan.Files, func(f File) bool { return f.Path == "config.json" })
	dest := filepath.Join(t.TempDir(), "checkpoint")
	h.mu.Lock()
	requests := h.requests
	h.mu.Unlock()
	if path, err := DownloadCheckpoint(t.Context(), plan, dest, nil); path != "" || err == nil || !strings.Contains(err.Error(), "no config.json") {
		t.Errorf("plan without config.json should fail: %q, %v", path, err)
	}
	for _, path := range []string{dest, dest + ".partial"} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("refused plan left %s: %v", path, err)
		}
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.requests != requests {
		t.Errorf("refused plan made %d network requests", h.requests-requests)
	}
}

func TestDownloadCheckpointRequiresWeights(t *testing.T) {
	const repo = "o/tiny"
	const missingShard = "model-00002-of-00002.safetensors"
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, repo)
	full, err := PlanCheckpoint(t.Context(), repo)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, wantError string
		keep            func(File) bool
	}{
		{"config only", "model.safetensors", func(f File) bool { return f.Path == "config.json" }},
		{"missing shard", missingShard, func(f File) bool { return f.Path != missingShard }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := full
			plan.Files = slices.DeleteFunc(slices.Clone(full.Files), func(f File) bool { return !tc.keep(f) })
			plan.Bytes = SetBytes(plan.Files)
			dest := filepath.Join(t.TempDir(), "checkpoint")
			if path, err := DownloadCheckpoint(t.Context(), plan, dest, nil); path != "" || err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Errorf("DownloadCheckpoint = %q, %v; want refusal mentioning %q", path, err, tc.wantError)
			}
			for _, path := range []string{dest, filepath.Join(dest+".partial", checkpointMarker)} {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Errorf("refused plan published %s: %v", path, err)
				}
			}
		})
	}
}

func TestCheckpointResumeUsesPlannedIndex(t *testing.T) {
	const repo = "o/tiny"
	const shard = "model-00001-of-00002.safetensors"
	const interrupted = "model-00002-of-00002.safetensors"
	for _, tc := range []struct {
		name                        string
		public, matching, malformed bool
	}{
		{name: "prepared with stale index"},
		{name: "prepared with malformed index", malformed: true},
		{name: "prepared with matching index", matching: true},
		{name: "public with stale index", public: true},
		{name: "public with malformed index", public: true, malformed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := shardedTinyRepo(t)
			original := files[checkpointIndex]
			stale := indexWithMissingTensor(t, original)
			if tc.matching {
				stale = original
			} else if tc.malformed {
				stale = bytes.Repeat([]byte("x"), len(original))
			}
			h := &fakeHF{files: files}
			h.serve(t, repo)
			plan, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			pin := plan.TreeDigest()
			if tc.public {
				data, err := json.Marshal(plan)
				if err != nil {
					t.Fatal(err)
				}
				var restored Plan
				if err := json.Unmarshal(data, &restored); err != nil {
					t.Fatal(err)
				}
				download = func(ctx context.Context, dest string, progress func(int64, int64, string)) (string, error) {
					return DownloadCheckpoint(ctx, restored, dest, progress)
				}
			} else {
				// A prepared download uses the bytes it planned even if upstream changes later.
				h.mu.Lock()
				h.files[checkpointIndex] = stale
				h.mu.Unlock()
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			staging := dest + ".partial"
			if err := os.Mkdir(staging, 0o755); err != nil {
				t.Fatal(err)
			}
			for name, raw := range map[string][]byte{checkpointIndex: stale, shard: files[shard], interrupted + ".part": files[interrupted][:32]} {
				if err := os.WriteFile(filepath.Join(staging, name), raw, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			f, err := os.Open(filepath.Join(staging, checkpointIndex))
			if err != nil {
				t.Fatal(err)
			}
			stagedIndex, err := f.Stat()
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if path, err := download(t.Context(), dest, nil); path != dest || err != nil {
				t.Fatalf("resume = %q, %v; want %s", path, err, dest)
			}
			if raw, err := os.ReadFile(filepath.Join(dest, checkpointIndex)); err != nil || !bytes.Equal(raw, original) {
				t.Errorf("published index differs from the planned bytes: %v", err)
			}
			if info, err := os.Stat(filepath.Join(dest, checkpointIndex)); err != nil || os.SameFile(stagedIndex, info) != tc.matching {
				t.Errorf("published index should reuse the staged file only when it matches: %v", err)
			}
			model, err := decoder.Load(dest, decoder.Options{Backend: "cpu"})
			if err != nil {
				t.Fatal(err)
			}
			model.Close()
			marker := readTestCheckpointMarker(t, dest)
			if i := slices.IndexFunc(marker.Files, func(f File) bool { return f.Path == checkpointIndex }); i < 0 || marker.Files[i].SHA256 != fmt.Sprintf("%x", sha256.Sum256(original)) {
				t.Error("marker lacks the digest of the planned index")
			}
			if i := slices.IndexFunc(plan.Files, func(f File) bool { return f.Path == checkpointIndex }); i < 0 || plan.Files[i].SHA256 != "" || plan.VerifyPin(pin) != nil {
				t.Error("the marker digest changed the public plan or the tree pin")
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			wantIndexHits := 1
			if tc.public {
				wantIndexHits++
			}
			if h.hits[checkpointIndex] != wantIndexHits || h.hits[shard] != 0 || h.hits[interrupted] != 1 || h.ranges[interrupted] != "bytes=32-" {
				t.Errorf("index should be fetched %d times, the completed shard reused, and the interrupted shard resumed: requests %v, ranges %v", wantIndexHits, h.hits, h.ranges)
			}
		})
	}
}

func indexWithMissingTensor(t *testing.T, original []byte) []byte {
	t.Helper()
	changed := bytes.Replace(original, []byte("model.embed_tokens.weight"), []byte("xodel.embed_tokens.weight"), 1)
	if bytes.Equal(changed, original) || len(changed) != len(original) || !json.Valid(changed) {
		t.Fatal("fixture must have two different valid indices of equal size")
	}
	return changed
}

func TestDownloadCheckpointRejectsInvalidRemoteIndexBeforeUpdatingStaging(t *testing.T) {
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	original := files[checkpointIndex]
	h := &fakeHF{files: files}
	h.serve(t, repo)
	plan, err := PlanCheckpoint(t.Context(), repo)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "checkpoint")
	staging := dest + ".partial"
	if err := os.Mkdir(staging, 0o755); err != nil {
		t.Fatal(err)
	}
	indexPath := filepath.Join(staging, checkpointIndex)
	if err := os.WriteFile(indexPath, original, 0o644); err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.files[checkpointIndex] = bytes.Repeat([]byte("x"), len(original))
	requests := h.requests
	h.mu.Unlock()
	if got, err := DownloadCheckpoint(t.Context(), plan, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "parsing "+checkpointIndex) {
		t.Errorf("refusing malformed remote index = %q, %v", got, err)
	}
	if raw, err := os.ReadFile(indexPath); err != nil || !bytes.Equal(raw, original) {
		t.Errorf("refusal changed the staged index: %v", err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.requests != requests+2 {
		t.Errorf("expected only config and index requests, got %d requests", h.requests-requests)
	}
}

func readTestCheckpointMarker(t *testing.T, dir string) markerDoc {
	t.Helper()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	marker, ok := readCheckpointMarker(root)
	if !ok {
		t.Fatal("failed to read checkpoint marker")
	}
	return marker
}

func TestCheckpointCompletedIndexCompatibility(t *testing.T) {
	const repo = "o/tiny"
	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			withCacheRoot(t)
			files := shardedTinyRepo(t)
			original := files[checkpointIndex]
			h := &fakeHF{files: files}
			h.serve(t, repo)
			ctx := t.Context()
			dest, err := CacheDir(repo)
			if err != nil {
				t.Fatal(err)
			}
			plan, download, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := download(ctx, dest, nil); err != nil {
				t.Fatal(err)
			}
			markerPath := filepath.Join(dest, checkpointMarker)
			if legacy {
				marker := readTestCheckpointMarker(t, dest)
				for i, f := range marker.Files {
					if f.Path == checkpointIndex {
						marker.Files[i].SHA256 = ""
					}
				}
				raw, err := json.Marshal(marker)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(markerPath, raw, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			marker, err := os.ReadFile(markerPath)
			if err != nil {
				t.Fatal(err)
			}
			h.mu.Lock()
			requests := h.requests
			h.mu.Unlock()
			if got, err := download(ctx, dest, nil); got != dest || err != nil {
				t.Errorf("matching plan should reuse the completed checkpoint: %q, %v", got, err)
			}
			if got, err := Resolve(ctx, "hf:"+repo+":safetensors", nil); got != dest || err != nil {
				t.Errorf("offline Resolve should accept the completed checkpoint: %q, %v", got, err)
			}
			changed := indexWithMissingTensor(t, original)
			h.mu.Lock()
			if h.requests != requests {
				t.Errorf("reusing the checkpoint made %d network requests", h.requests-requests)
			}
			h.files[checkpointIndex] = changed
			h.mu.Unlock()
			updatedPlan, updatedDownload, err := PrepareCheckpointFor(ctx, repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := updatedPlan.VerifyPin(plan.TreeDigest()); err != nil {
				t.Fatalf("same-size non-LFS index changed the tree pin: %v", err)
			}
			if got, err := updatedDownload(ctx, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "index differs from the current checkpoint plan") {
				t.Errorf("changed index should refuse the completed destination: %q, %v", got, err)
			}
			indexPath := filepath.Join(dest, checkpointIndex)
			if raw, err := os.ReadFile(indexPath); err != nil || !bytes.Equal(raw, original) {
				t.Errorf("refusal changed the completed index: %v", err)
			}
			if raw, err := os.ReadFile(markerPath); err != nil || !bytes.Equal(raw, marker) {
				t.Errorf("reusing or refusing the destination changed the marker: %v", err)
			}
			if !legacy {
				if err := os.WriteFile(indexPath, changed, 0o644); err != nil {
					t.Fatal(err)
				}
				if _, ok := CachedCheckpoint(dest); ok {
					t.Error("CachedCheckpoint accepted a same-size index that differs from the marker digest")
				}
				if got, err := Resolve(ctx, "hf:"+repo+":safetensors", nil); got != "" || err == nil || !strings.Contains(err.Error(), "index does not match checkpoint metadata") {
					t.Errorf("offline Resolve should refuse the changed index: %q, %v", got, err)
				}
			}
		})
	}
}

func TestCachedCheckpointWeightLayout(t *testing.T) {
	const repo = "o/tiny"
	const single = "model.safetensors"
	const index = "model.safetensors.index.json"
	const shard = "model-00001-of-00002.safetensors"
	const otherShard = "model-00002-of-00002.safetensors"
	fixture := shardedTinyRepo(t)
	weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	fixture[single] = weights
	shardIndex := string(fixture[index])
	for _, tc := range []struct {
		name, indexJSON                 string
		weights                         []string
		unlisted, missing, absoluteLink string
		outsideLink, wantComplete       bool
	}{
		{name: "legacy single", weights: []string{single}, wantComplete: true},
		{name: "legacy sharded", indexJSON: shardIndex, weights: []string{shard, otherShard}, wantComplete: true},
		{name: "legacy sharded with local index link", indexJSON: shardIndex, weights: []string{shard, otherShard}, absoluteLink: index, wantComplete: true},
		{name: "legacy single with local weight link", weights: []string{single}, absoluteLink: single, wantComplete: true},
		{name: "external index link", indexJSON: shardIndex, weights: []string{shard, otherShard}, absoluteLink: index, outsideLink: true},
		{name: "external weight link", weights: []string{single}, absoluteLink: single, outsideLink: true},
		{name: "config only"},
		{name: "unlisted shard", indexJSON: shardIndex, weights: []string{shard}, unlisted: otherShard},
		{name: "missing shard on disk", indexJSON: shardIndex, weights: []string{shard, otherShard}, missing: otherShard},
		{name: "malformed index", indexJSON: `{"weight_map":`, weights: []string{shard}},
		{name: "empty index", indexJSON: `{"weight_map":{}}`},
		{name: "unsafe shard", indexJSON: `{"weight_map":{"first":"../outside.safetensors"}}`},
		{name: "index references config as a shard", indexJSON: `{"weight_map":{"first":"config.json"}}`},
		{name: "unlisted index shadows single file", weights: []string{single}, unlisted: index},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withCacheRoot(t)
			dir, err := CacheDir(repo)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			body := map[string][]byte{"config.json": fixture["config.json"]}
			for _, name := range tc.weights {
				body[name] = fixture[name]
			}
			if tc.indexJSON != "" {
				body[index] = []byte(tc.indexJSON)
			}
			marker := markerDoc{Repo: repo}
			for name, raw := range body {
				marker.Files = append(marker.Files, File{Path: name, Size: int64(len(raw))})
			}
			raw, err := json.Marshal(marker)
			if err != nil {
				t.Fatal(err)
			}
			body[checkpointMarker] = raw
			if tc.unlisted != "" {
				body[tc.unlisted] = fixture[tc.unlisted]
			}
			delete(body, tc.missing)
			for name, raw := range body {
				path := filepath.Join(dir, name)
				if name == tc.absoluteLink {
					target := path + ".data"
					if tc.outsideLink {
						target = filepath.Join(t.TempDir(), name)
					}
					saveCheckpointConfigLink(t, target, path)
					path = target
				}
				if err := os.WriteFile(path, raw, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			h := &fakeHF{}
			h.serve(t, repo)
			if path, ok := CachedCheckpoint(dir); ok != tc.wantComplete || (path == dir) != tc.wantComplete {
				t.Errorf("CachedCheckpoint = %q, %v; want complete=%v", path, ok, tc.wantComplete)
			}
			if path, err := Resolve(t.Context(), "hf:"+repo+":safetensors", nil); (err == nil) != tc.wantComplete || (path == dir) != tc.wantComplete {
				t.Errorf("Resolve = %q, %v; want complete=%v", path, err, tc.wantComplete)
			}
			if tc.wantComplete {
				model, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
				if err != nil {
					t.Fatal(err)
				}
				model.Close()
			}
			entries, err := CacheEntries()
			if err != nil || len(entries) != 1 || entries[0].Complete != tc.wantComplete {
				t.Errorf("CacheEntries = %+v, %v; want one entry with complete=%v", entries, err, tc.wantComplete)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.requests != 0 {
				t.Errorf("cache lookup made %d network requests", h.requests)
			}
		})
	}
}

func TestCachedCheckpointRequiresConfigEntry(t *testing.T) {
	dir := t.TempDir()
	files := map[string][]byte{
		"config.json":       []byte(`{"model_type":"llama"}`),
		"model.safetensors": []byte("weight"),
		checkpointMarker:    []byte(`{"repo":"o/tiny","files":[{"Path":"model.safetensors","Size":6}]}`),
	}
	for name, raw := range files {
		if err := os.WriteFile(filepath.Join(dir, name), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := CachedCheckpoint(dir); ok {
		t.Fatal("a marker without config.json claimed a usable checkpoint")
	}
}

func TestDownloadCheckpointRevalidatesRestoredConfigPlan(t *testing.T) {
	const repo = "o/tiny"
	const shardPart = "model-00001-of-00002.safetensors.part"
	files := shardedTinyRepo(t)
	original := files["config.json"]
	h := &fakeHF{files: files}
	h.serve(t, repo)
	ctx := t.Context()
	oldPlan, err := PlanCheckpoint(ctx, repo)
	if err != nil {
		t.Fatal(err)
	}
	updated := configWithDifferentEOS(t, original)
	h.mu.Lock()
	h.files["config.json"] = updated
	h.mu.Unlock()
	plan, err := PlanCheckpoint(ctx, repo)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	type request struct {
		Plan
		Destination string
	}
	const destination = "models/tiny"
	embedded, err := json.Marshal(request{Plan: plan, Destination: destination})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name        string
		initial     Plan
		embedded    bool
		unavailable bool
	}{
		{name: "zero receiver"},
		{name: "receiver with an older config", initial: oldPlan},
		{name: "embedded receiver with an older config", initial: oldPlan, embedded: true},
		{name: "config unavailable", initial: oldPlan, unavailable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restored := tc.initial
			if tc.embedded {
				got := request{Plan: restored}
				if err := json.Unmarshal(embedded, &got); err != nil {
					t.Fatal(err)
				}
				if got.Destination != destination {
					t.Errorf("restored destination = %q; want %q", got.Destination, destination)
				}
				restored = got.Plan
			} else if err := json.Unmarshal(data, &restored); err != nil {
				t.Fatal(err)
			}
			if err := restored.VerifyPin(plan.TreeDigest()); err != nil {
				t.Fatalf("restored metadata lost its tree pin: %v", err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			staging := dest + ".partial"
			if err := os.Mkdir(staging, 0o755); err != nil {
				t.Fatal(err)
			}
			staged := map[string][]byte{"config.json": original, shardPart: files["model-00001-of-00002.safetensors"][:32]}
			for name, raw := range staged {
				if err := os.WriteFile(filepath.Join(staging, name), raw, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			h.mu.Lock()
			requests := h.requests
			configHits := h.hits["config.json"]
			if tc.unavailable {
				delete(h.files, "config.json")
			}
			h.mu.Unlock()
			got, err := DownloadCheckpoint(t.Context(), restored, dest, nil)
			if tc.unavailable {
				if err == nil || got != "" {
					t.Errorf("unavailable config should refuse the download: %q, %v", got, err)
				}
				if _, err := os.Stat(dest); !os.IsNotExist(err) {
					t.Errorf("restored plan published a checkpoint: %v", err)
				}
				for name, want := range staged {
					if got, err := os.ReadFile(filepath.Join(staging, name)); err != nil || !bytes.Equal(got, want) {
						t.Errorf("refusal changed staged %s: %v", name, err)
					}
				}
				h.mu.Lock()
				defer h.mu.Unlock()
				if h.requests != requests+1 {
					t.Errorf("expected only the failed config request, got %d requests", h.requests-requests)
				}
				return
			}
			if err != nil || got != dest {
				t.Fatalf("download restored plan: %q, %v", got, err)
			}
			if raw, err := os.ReadFile(filepath.Join(dest, "config.json")); err != nil || !bytes.Equal(raw, updated) {
				t.Errorf("download used stale config: %v", err)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.hits["config.json"] != configHits+1 || h.ranges["model-00001-of-00002.safetensors"] != "bytes=32-" {
				t.Errorf("expected one config fetch and a resumed shard: hits %v, ranges %v", h.hits, h.ranges)
			}
		})
	}
}

// TestCheckpoint_unsupportedDeclinedBeforeAnyWeight: a model_type this build cannot load is refused after reading
// config.json and before any weight file is requested.
func TestCheckpoint_unsupportedDeclinedBeforeAnyWeight(t *testing.T) {
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	files["config.json"] = []byte(`{"model_type": "made_up_arch", "torch_dtype": "bfloat16"}`)
	h := &fakeHF{files: files, gated: false}
	h.serve(t, "o/tiny")
	_, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil)
	if err == nil || !strings.Contains(err.Error(), "made_up_arch") {
		t.Fatalf("an unsupported model_type: err %v, want a refusal naming it", err)
	}
	for p, c := range h.hits {
		if p != "config.json" && c > 0 {
			t.Fatalf("%s was requested (%d) before the refusal", p, c)
		}
	}
}

// TestCheckpoint_gatedDeclinedBeforeAnyRequest: a gated original (the doc's §2, option (c): anonymous only) is refused
// by CheckAccess before the tree or any file is read.
func TestCheckpoint_gatedDeclinedBeforeAnyRequest(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: "manual"}
	h.serve(t, "o/tiny")
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil || !strings.Contains(err.Error(), "gated") {
		t.Fatalf("a gated repo: err %v, want the gated refusal", err)
	}
	if len(h.hits) != 0 {
		t.Fatalf("a gated repo had files requested: %v", h.hits)
	}
}

// TestCheckpoint_sizeNoteOnFullPrecision: the plan states a bf16 original's download cost before the transfer.
func TestCheckpoint_sizeNoteOnFullPrecision(t *testing.T) {
	p := Plan{Files: []File{{Path: "model.safetensors", Size: 15e9}}, Bytes: 15e9, Dtype: "bfloat16"}
	if n := p.SizeNote(); !strings.Contains(n, "full-precision") || !strings.Contains(n, "quarter") {
		t.Fatalf("size note for a bf16 original: %q", n)
	}
	if n := (Plan{Files: []File{{Size: 1}}, Bytes: 1, Dtype: "int8"}).SizeNote(); strings.Contains(n, "full-precision") {
		t.Fatalf("a non-full-precision plan carries the full-precision warning: %q", n)
	}
}

// TestCheckpoint_badIndexAndPathsRefused: a shard index naming a file the repo lacks, and an unsafe path, are refused
// at plan time.
func TestCheckpoint_badIndexAndPathsRefused(t *testing.T) {
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	delete(files, "model-00002-of-00002.safetensors")
	h := &fakeHF{files: files, gated: false}
	h.serve(t, "o/tiny")
	if _, err := PlanCheckpoint(context.Background(), "o/tiny"); err == nil || !strings.Contains(err.Error(), "model-00002-of-00002.safetensors") {
		t.Fatalf("an index naming a missing shard: err %v", err)
	}
	for _, p := range []string{"../x", "/abs", "a/../b", "a\\b", ""} {
		if safeRepoPath(p) {
			t.Fatalf("safeRepoPath(%q) = true", p)
		}
	}
	if !safeRepoPath("vision/model.safetensors") {
		t.Fatal("a nested relative path was refused")
	}
}

// TestCheckpoint_existingNonCheckpointRefused: a directory at the destination that goinfer did not complete is refused,
// not overwritten.
func TestCheckpoint_existingNonCheckpointRefused(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t), gated: false}
	h.serve(t, "o/tiny")
	dir, _ := CacheDir("o/tiny")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mine.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Resolve(context.Background(), "hf:o/tiny:safetensors", nil); err == nil || !strings.Contains(err.Error(), "not a complete checkpoint") {
		t.Fatalf("an existing foreign directory: err %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "mine.txt")); string(b) != "x" {
		t.Fatal("the existing directory's file was touched")
	}
}

// TestPlanCheckpointFor_embeddingEncoder is P7's plan half: an embedding encoder's repo (a NomicBert, which no generative
// family loads) is declined by the generative plan, planned by the caller's own check, and the plan carries the
// sentence-transformers pooling config its encoder reads (1_Pooling/config.json), which the fetch writes in its folder. A
// model_type the caller's check refuses is declined with only config.json read.
func TestPlanCheckpointFor_embeddingEncoder(t *testing.T) {
	withCacheRoot(t)
	files := map[string][]byte{
		"config.json":           []byte(`{"model_type":"nomic_bert","torch_dtype":"float32"}`),
		"model.safetensors":     []byte("weights"),
		"tokenizer.json":        []byte(`{}`),
		"modules.json":          []byte(`[]`),
		"1_Pooling/config.json": []byte(`{"pooling_mode_cls_token":true}`),
		"README.md":             []byte("# readme"),
	}
	h := &fakeHF{files: files, gated: false}
	h.serve(t, "o/enc")
	if _, err := PlanCheckpoint(context.Background(), "o/enc"); err == nil || !strings.Contains(err.Error(), "nomic_bert") {
		t.Fatalf("the generative plan on a NomicBert: %v, want a decline naming it", err)
	}
	encoderLoads := func(mt string) (string, error) {
		if mt == "nomic_bert" {
			return "nomic_bert encoder", nil
		}
		return "", fmt.Errorf("model_type %q, which the encoder does not load", mt)
	}
	p, err := PlanCheckpointFor(context.Background(), "o/enc", encoderLoads)
	if err != nil {
		t.Fatal(err)
	}
	var paths []string
	for _, f := range p.Files {
		paths = append(paths, f.Path)
	}
	if got, want := strings.Join(paths, " "), "1_Pooling/config.json config.json model.safetensors modules.json tokenizer.json"; got != want || p.Family != "nomic_bert encoder" {
		t.Fatalf("plan %q family %q; want %q", got, p.Family, want)
	}
	dir, err := ResolveCheckpointFor(context.Background(), "hf:o/enc:safetensors", encoderLoads)
	if err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "1_Pooling", "config.json")); err != nil || !strings.Contains(string(b), "cls") {
		t.Fatalf("the pooling config in the fetched set: %q, %v", b, err)
	}

	files["config.json"] = []byte(`{"model_type":"llama"}`)
	h2 := &fakeHF{files: files, gated: false}
	h2.serve(t, "o/notenc")
	if _, err := ResolveCheckpointFor(context.Background(), "hf:o/notenc:safetensors", encoderLoads); err == nil || !strings.Contains(err.Error(), "does not load") {
		t.Fatalf("a model_type the caller refuses: %v, want its refusal", err)
	}
	if h2.hits["model.safetensors"] != 0 {
		t.Fatalf("the refused repo's weights were fetched %d times", h2.hits["model.safetensors"])
	}
}

// TestAnyLoads_generativeOrEncoder: the pull CLI's check takes a generative family or the embedding encoder, and refuses
// anything else naming both (task-checkpoint-fetch P7's last gap: `pull <encoder>:safetensors` declined what serve
// --embed-model fetches itself). PlanCheckpoint, --model's and the web UI's check, still refuses the encoder.
func TestAnyLoads_generativeOrEncoder(t *testing.T) {
	if fam, err := AnyLoads("llama"); err != nil || fam == "" || fam == EncoderFamily {
		t.Errorf("llama: %q, %v; want a generative family", fam, err)
	}
	if fam, err := AnyLoads("nomic_bert"); err != nil || fam != EncoderFamily {
		t.Errorf("nomic_bert: %q, %v; want %q", fam, err, EncoderFamily)
	}
	if _, err := AnyLoads("made_up_arch"); err == nil || !strings.Contains(err.Error(), "made_up_arch") || !strings.Contains(err.Error(), "nomic_bert") {
		t.Errorf("made_up_arch: %v; want a refusal naming it and the encoder", err)
	}
	if _, err := generativeLoads("nomic_bert"); err == nil {
		t.Error("the generative check accepts nomic_bert: --model and the web UI would offer an encoder they cannot run")
	}

	withCacheRoot(t)
	h := &fakeHF{files: map[string][]byte{
		"config.json":           []byte(`{"model_type":"nomic_bert","torch_dtype":"float32"}`),
		"model.safetensors":     []byte("weights"),
		"tokenizer.json":        []byte(`{}`),
		"1_Pooling/config.json": []byte(`{"pooling_mode_cls_token":true}`),
	}, gated: false}
	h.serve(t, "o/enc")
	p, err := PlanCheckpointFor(context.Background(), "o/enc", AnyLoads)
	if err != nil || p.Family != EncoderFamily {
		t.Fatalf("the CLI's plan of a NomicBert: family %q, %v; want %q", p.Family, err, EncoderFamily)
	}
}

func TestPlan_treeDigestFollowsTheFiles(t *testing.T) {
	base := Plan{Repo: "o/r", Files: []File{{Path: "config.json", Size: 10}, {Path: "model.safetensors", Size: 1000, SHA256: "aa"}}}
	d := base.TreeDigest()
	if len(d) != 64 {
		t.Fatalf("digest %q is not 64 hex characters", d)
	}
	reordered := Plan{Repo: "o/r", Files: []File{base.Files[1], base.Files[0]}}
	if reordered.TreeDigest() != d {
		t.Error("the digest depends on the listing order: it must sort by path")
	}
	for name, mut := range map[string]func(p *Plan){
		"a weight's digest": func(p *Plan) { p.Files[1].SHA256 = "ab" },
		"a weight's size":   func(p *Plan) { p.Files[1].Size++ },
		"a config's size":   func(p *Plan) { p.Files[0].Size++ },
		"a file's name":     func(p *Plan) { p.Files[0].Path = "config2.json" },
		"an added file":     func(p *Plan) { p.Files = append(p.Files, File{Path: "extra.json", Size: 1}) },
	} {
		q := Plan{Repo: "o/r", Files: append([]File(nil), base.Files...)}
		mut(&q)
		if q.TreeDigest() == d {
			t.Errorf("changing %s did not change the tree digest", name)
		}
	}
}

func TestPlan_verifyPin(t *testing.T) {
	p := Plan{Repo: "o/r", Files: []File{{Path: "model.safetensors", Size: 5, SHA256: "aa"}}}
	if err := p.VerifyPin(""); err != nil {
		t.Errorf("an empty pin (the explicit form) must pin nothing: %v", err)
	}
	if err := p.VerifyPin(p.TreeDigest()); err != nil {
		t.Errorf("the matching pin was refused: %v", err)
	}
	err := p.VerifyPin(strings.Repeat("0", 64))
	if err == nil {
		t.Fatal("a pin that does not match the plan was accepted: a re-uploaded checkpoint would ride under the recommended name")
	}
	if !strings.Contains(err.Error(), "nothing was downloaded") || !strings.Contains(err.Error(), CheckpointSelector) {
		t.Errorf("the refusal does not say that nothing was downloaded and how to take the new files: %v", err)
	}
}
