package pull

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCheckpointProtectsUnownedPlannedPartials(t *testing.T) {
	const repo = "o/tiny"
	const shard = "model-00001-of-00002.safetensors"
	const tokenizer = "tokenizer.json"
	for _, tc := range []struct {
		name         string
		file         string
		swap, replan bool
	}{
		{name: "unrecorded shard on layout change", file: shard, replan: true},
		{name: "replaced shard on resume", file: shard, swap: true},
		{name: "unrecorded tokenizer on layout change", file: tokenizer, replan: true},
		{name: "replaced tokenizer on resume", file: tokenizer, swap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			split := shardedTinyRepo(t)
			split[tokenizer] = bytes.Repeat([]byte("x"), 8192)
			h := &fakeHF{files: split, failOnce: tc.file}
			if tc.replan {
				weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
				if err != nil {
					t.Fatal(err)
				}
				h.files = map[string][]byte{"config.json": split["config.json"], checkpointWeights: weights}
				h.failOnce = checkpointWeights
			}
			h.serve(t, repo)
			dest := filepath.Join(t.TempDir(), "checkpoint")
			_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := download(t.Context(), dest, nil); err == nil {
				t.Fatal("expected interruption")
			}
			part := filepath.Join(testCheckpointStagePath(dest), tc.file+".part")
			data := []byte("local data")
			if tc.swap {
				info, err := os.Stat(part)
				if err != nil {
					t.Fatal(err)
				}
				data = bytes.Repeat([]byte("X"), int(info.Size()))
				if err := os.WriteFile(part+".new", data, 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(part+".new", info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(part+".new", part); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(part, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if tc.replan {
				h.mu.Lock()
				h.files = split
				h.mu.Unlock()
				_, download, err = PrepareCheckpointFor(t.Context(), repo, nil)
				if err != nil {
					t.Fatal(err)
				}
			}
			h.mu.Lock()
			hits := h.hits[tc.file]
			h.mu.Unlock()
			if got, err := download(t.Context(), dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "verified transfer record") {
				t.Errorf("pull with an unowned partial = %q, %v", got, err)
			}
			if raw, err := os.ReadFile(part); err != nil || !bytes.Equal(raw, data) {
				t.Errorf("unowned partial changed: %d bytes, %v", len(raw), err)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.hits[tc.file] != hits {
				t.Errorf("fetched %s with an unowned partial", tc.file)
			}
		})
	}
}

func TestCheckpointProtectsPartialsChangedDuringRequest(t *testing.T) {
	const repo = "o/tiny"
	const shard = "model-00001-of-00002.safetensors"
	const tokenizer = "tokenizer.json"
	for _, tc := range []struct {
		name  string
		file  string
		owned bool
	}{
		{name: "created during full request", file: tokenizer},
		{name: "replaced during full request", file: tokenizer, owned: true},
		{name: "replaced during range request", file: shard, owned: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := shardedTinyRepo(t)
			files[tokenizer] = bytes.Repeat([]byte("x"), 8192)
			h := &fakeHF{files: files}
			h.serve(t, repo)
			_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			part := tc.file + ".part"
			var partials []testCheckpointPartial
			if tc.owned {
				partials = append(partials, testCheckpointPartial{part, files[tc.file][:32]})
			}
			stage := createTestCheckpointStage(t, dest, partials...)
			next := filepath.Join(stage, "replacement")
			want := []byte("keep this inode")
			if err := os.WriteFile(next, want, 0o600); err != nil {
				t.Fatal(err)
			}
			h.mu.Lock()
			h.beforeSend = func(name string) {
				if name == tc.file {
					if err := os.Rename(next, filepath.Join(stage, part)); err != nil {
						t.Errorf("replacing partial during request: %v", err)
					}
				}
			}
			h.mu.Unlock()
			if got, err := download(t.Context(), dest, nil); got != "" || err == nil {
				t.Errorf("pull with a concurrent partial change = %q, %v", got, err)
			}
			if raw, err := os.ReadFile(filepath.Join(stage, part)); err != nil || !bytes.Equal(raw, want) {
				t.Errorf("changed partial was lost: %q, %v", raw, err)
			}
		})
	}
}

func TestCheckpointRetriesCleanupAfterKilledPublication(t *testing.T) {
	const child = "GOINFER_TEST_CHECKPOINT_PUBLISHED"
	const repo = "o/tiny"
	const obsolete = "model-00001-of-00002.safetensors"
	const interrupted = "model-00002-of-00002.safetensors"
	const unknown = "notes.txt"
	if dest := os.Getenv(child); dest != "" {
		hfAPI = os.Getenv("GOINFER_TEST_CHECKPOINT_API")
		hfCDN = os.Getenv("GOINFER_TEST_CHECKPOINT_CDN")
		renameCheckpointPublication = func(root *os.Root, src, dest string) error {
			if err := root.Rename(src, dest); err != nil {
				return err
			}
			fmt.Println("published")
			_, err := io.Copy(io.Discard, os.Stdin)
			return err
		}
		_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = download(t.Context(), dest, nil)
		t.Fatalf("expected the process to be killed after rename: %v", err)
	}
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files, failOnce: interrupted}
	h.serve(t, repo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := download(t.Context(), dest, nil); err == nil || !strings.Contains(err.Error(), interrupted) {
		t.Fatalf("expected interrupted shard, got %v", err)
	}
	stage := testCheckpointStagePath(dest)
	if err := os.WriteFile(filepath.Join(stage, unknown), []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	single, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.files = map[string][]byte{"config.json": files["config.json"], checkpointWeights: single}
	h.mu.Unlock()
	_, download, err = PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointRetriesCleanupAfterKilledPublication$")
	cmd.Env = append(os.Environ(), child+"="+dest, "GOINFER_TEST_CHECKPOINT_API="+hfAPI, "GOINFER_TEST_CHECKPOINT_CDN="+hfCDN)
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	if line, err := bufio.NewReader(out).ReadString('\n'); line != "published\n" || err != nil {
		t.Fatalf("publication did not reach rename: %q, %v", line, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Fatal("checkpoint was incomplete at the crash boundary")
	}
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("reuse during the live publisher = %q, %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(stage, obsolete)); err != nil {
		t.Fatalf("staging changed while publisher held its lock: %v", err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("publisher exited normally")
	}
	before := dirBytes(filepath.Dir(stage))
	h.mu.Lock()
	hits := h.hits[checkpointWeights]
	h.mu.Unlock()
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("retry after SIGKILL = %q, %v", got, err)
	}
	for _, name := range []string{obsolete, interrupted + ".part", checkpointWeights} {
		if _, err := os.Lstat(filepath.Join(stage, name)); !os.IsNotExist(err) {
			t.Errorf("owned file %s remains after retry: %v", name, err)
		}
	}
	if receipts, err := filepath.Glob(filepath.Join(filepath.Dir(stage), checkpointReceiptPrefix+"*")); err != nil || len(receipts) != 0 {
		t.Errorf("transfer receipts remain: %v, %v", receipts, err)
	}
	if raw, err := os.ReadFile(filepath.Join(stage, unknown)); err != nil || string(raw) != "keep" {
		t.Errorf("unowned data changed: %q, %v", raw, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Fatal("cleanup damaged the completed checkpoint")
	}
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("reuse after recovery = %q, %v", got, err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.hits[checkpointWeights] != hits {
		t.Error("cleanup re-fetched weights after publication")
	}
	t.Logf("workspace bytes after post-publication crash: %d → %d", before, dirBytes(filepath.Dir(stage)))
}
