package pull

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCheckpointRejectsReplacedStagedWeight(t *testing.T) {
	const repo = "o/tiny"
	const shard = "model-00001-of-00002.safetensors"
	const last = "tokenizer_config.json"
	for _, duringPublication := range []bool{false, true} {
		t.Run(fmt.Sprintf("during publication=%t", duringPublication), func(t *testing.T) {
			files := shardedTinyRepo(t)
			files[last] = []byte(`{"model_max_length":2048}`)
			h := &fakeHF{files: files}
			h.serve(t, repo)
			_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			replaced := false
			replace := func() {
				file := filepath.Join(testCheckpointStagePath(dest), shard)
				info, err := os.Stat(file)
				if err != nil {
					t.Fatal(err)
				}
				bad := bytes.Clone(files[shard])
				bad[len(bad)-1] ^= 1
				tmp := file + ".replacement"
				if err := os.WriteFile(tmp, bad, 0o644); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(tmp, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(tmp, file); err != nil {
					t.Fatal(err)
				}
				replaced = true
			}
			if duringPublication {
				_, err := download(t.Context(), dest, func(done, total int64, name string) {
					if name == last && done == total && !replaced {
						replace()
					}
				})
				if !replaced || err == nil || !strings.Contains(err.Error(), "digest") {
					t.Errorf("expected rejection of the staged replacement, got %v", err)
				}
				if _, err := os.Lstat(dest); !os.IsNotExist(err) {
					t.Errorf("published corrupt data: %v", err)
				}
			} else {
				h.mu.Lock()
				h.failOnce = last
				h.mu.Unlock()
				if _, err := download(t.Context(), dest, nil); err == nil {
					t.Fatal("expected interruption")
				}
				replace()
				if _, err := download(t.Context(), dest, nil); err != nil {
					t.Fatal(err)
				}
				if raw, err := os.ReadFile(filepath.Join(dest, shard)); err != nil || !bytes.Equal(raw, files[shard]) {
					t.Errorf("resumed pull kept replacement bytes: %v", err)
				}
				h.mu.Lock()
				hits := h.hits[shard]
				h.mu.Unlock()
				if hits != 2 {
					t.Errorf("replaced shard fetched %d times, want 2", hits)
				}
				if _, ok := CachedCheckpoint(dest); !ok {
					t.Error("the repaired checkpoint did not verify")
				}
			}
		})
	}
}

func TestCheckpointUnidentifiedRetriesRetireTransferReceipts(t *testing.T) {
	const repo = "o/tiny"
	const name = "tokenizer.json"
	const last = "tokenizer_config.json"
	files := shardedTinyRepo(t)
	files[last] = []byte(`{"model_max_length":2048}`)
	files[name] = bytes.Repeat([]byte("a"), 64<<10)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	var before int64
	for i := range 4 {
		h.mu.Lock()
		h.failOnce = last
		h.files[name] = bytes.Repeat([]byte{byte('a' + i)}, 64<<10)
		h.mu.Unlock()
		if _, err := download(t.Context(), dest, nil); err == nil || !strings.Contains(err.Error(), last) {
			t.Fatalf("expected retry interruption, got %v", err)
		}
		bytesOnDisk := dirBytes(filepath.Dir(testCheckpointStagePath(dest)))
		if i == 0 {
			before = bytesOnDisk
		} else if bytesOnDisk > before+4096 {
			t.Errorf("attempt %d retained old receipt bytes: %d → %d", i+1, before, bytesOnDisk)
		}
	}
	t.Logf("workspace bytes after retries %d → %d", before, dirBytes(filepath.Dir(testCheckpointStagePath(dest))))
}

func TestCheckpointReclaimsObsoletePartialAndPreservesUnknownPartial(t *testing.T) {
	const repo = "o/tiny"
	const interrupted = "model-00001-of-00002.safetensors"
	const unknown = "model-00002-of-00002.safetensors.part"
	const child = "GOINFER_TEST_CHECKPOINT_PARTIAL_OUTPUT"
	if dest := os.Getenv(child); dest != "" {
		hfAPI = os.Getenv("GOINFER_TEST_CHECKPOINT_API")
		hfCDN = os.Getenv("GOINFER_TEST_CHECKPOINT_CDN")
		_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = download(t.Context(), dest, nil)
		t.Fatalf("transfer returned before the process was killed: %v", err)
	}
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files, failOnce: interrupted, pausePartial: true}
	h.serve(t, repo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointReclaimsObsoletePartialAndPreservesUnknownPartial$")
	cmd.Env = append(os.Environ(), child+"="+dest, "GOINFER_TEST_CHECKPOINT_API="+hfAPI, "GOINFER_TEST_CHECKPOINT_CDN="+hfCDN)
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	staging := testCheckpointStagePath(dest)
	partial := filepath.Join(staging, interrupted+".part")
	var partialBytes int64
	for partialBytes == 0 {
		if info, err := os.Stat(partial); err == nil {
			partialBytes = info.Size()
		}
		select {
		case <-ctx.Done():
			_ = cmd.Wait()
			t.Fatalf("child never wrote the partial: %s", output.String())
		case <-time.After(10 * time.Millisecond):
		}
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil || output.Len() != 0 {
		t.Fatalf("expected the writer to be killed: %v\n%s", err, output.String())
	}
	if partialBytes >= int64(len(files[interrupted])) {
		t.Fatalf("fixture completed the shard: %d bytes", partialBytes)
	}
	if err := os.WriteFile(filepath.Join(staging, unknown), []byte("untracked"), 0o644); err != nil {
		t.Fatal(err)
	}
	single, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.files = map[string][]byte{"config.json": files["config.json"], checkpointWeights: single}
	h.mu.Unlock()
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := download(t.Context(), dest, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(partial); !os.IsNotExist(err) {
		t.Errorf("%d obsolete partial bytes were kept: %v", partialBytes, err)
	}
	t.Logf("obsolete partial contained %d bytes when the writer was killed", partialBytes)
	if raw, err := os.ReadFile(filepath.Join(staging, unknown)); err != nil || string(raw) != "untracked" {
		t.Errorf("untracked partial changed: %q, %v", raw, err)
	}
}
