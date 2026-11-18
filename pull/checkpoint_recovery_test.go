package pull

import (
	"bufio"
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

func TestCheckpointRecoversWhenInitializerIsKilled(t *testing.T) {
	const child = "GOINFER_TEST_CHECKPOINT_INIT_DIR"
	const kind = "GOINFER_TEST_CHECKPOINT_INIT_KIND"
	const destName = "checkpoint"
	if dir := os.Getenv(child); dir != "" {
		parent, err := os.OpenRoot(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer parent.Close()
		mkdir := mkdirCheckpointInit
		mkdirCheckpointInit = func(root *os.Root, name string) error {
			if err := mkdir(root, name); err != nil {
				return err
			}
			stage := filepath.Base(root.Name()) == checkpointWorkspaceName(destName)
			if stage == (os.Getenv(kind) == "stage") {
				fmt.Println(filepath.Join(root.Name(), name))
				_, _ = io.Copy(io.Discard, os.Stdin)
			}
			return nil
		}
		work, err := lockCheckpointDestination(parent, destName)
		if err != nil {
			t.Fatal(err)
		}
		defer work.Close()
		stage, err := work.openStage(true)
		if err != nil {
			t.Fatal(err)
		}
		stage.Close()
		return
	}
	for _, mode := range []string{"workspace", "stage"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			dest := filepath.Join(dir, destName)
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointRecoversWhenInitializerIsKilled$")
			cmd.Env = append(os.Environ(), child+"="+dir, kind+"="+mode)
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
			t.Cleanup(func() {
				if cmd.ProcessState == nil {
					_ = cmd.Process.Kill()
					_ = cmd.Wait()
				}
			})
			line, err := bufio.NewReader(out).ReadString('\n')
			if err != nil {
				t.Fatalf("initializer did not reach mkdir: %q, %v", line, err)
			}
			tmp := strings.TrimSpace(line)
			if !strings.HasPrefix(filepath.Base(tmp), ".goinfer-init-") {
				t.Fatalf("unexpected temporary directory %q", tmp)
			}
			canonical := filepath.Join(dir, checkpointWorkspaceName(destName))
			if mode == "stage" {
				canonical = filepath.Join(canonical, checkpointStageName)
			}
			if _, err := os.Lstat(canonical); !os.IsNotExist(err) {
				t.Fatalf("canonical directory exists before its owner: %v", err)
			}
			if err := cmd.Process.Kill(); err != nil {
				t.Fatal(err)
			}
			if err := cmd.Wait(); err == nil {
				t.Fatal("initializer exited normally")
			}
			if err := os.WriteFile(filepath.Join(tmp, "notes.txt"), []byte("keep"), 0o644); err != nil {
				t.Fatal(err)
			}
			const repo = "o/tiny"
			h := &fakeHF{files: shardedTinyRepo(t)}
			h.serve(t, repo)
			_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
				t.Fatalf("retry after SIGKILL = %q, %v", got, err)
			}
			if raw, err := os.ReadFile(filepath.Join(tmp, "notes.txt")); err != nil || string(raw) != "keep" {
				t.Errorf("retry changed the abandoned directory: %q, %v", raw, err)
			}
		})
	}
}

func TestCheckpointReclaimsOldWeightsAcrossProcesses(t *testing.T) {
	const child = "GOINFER_TEST_CHECKPOINT_INTERRUPTED_OUTPUT"
	const repo = "o/tiny"
	const tokenizer = "tokenizer_config.json"
	if dest := os.Getenv(child); dest != "" {
		hfAPI = os.Getenv("GOINFER_TEST_CHECKPOINT_API")
		hfCDN = os.Getenv("GOINFER_TEST_CHECKPOINT_CDN")
		_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := download(t.Context(), dest, nil); err == nil || !strings.Contains(err.Error(), tokenizer) {
			t.Fatalf("expected tokenizer interruption, got %v", err)
		}
		return
	}
	files := shardedTinyRepo(t)
	files[tokenizer] = []byte(`{"model_max_length":2048}`)
	h := &fakeHF{files: files, failOnce: tokenizer}
	h.serve(t, repo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCheckpointReclaimsOldWeightsAcrossProcesses$")
	cmd.Env = append(os.Environ(), child+"="+dest, "GOINFER_TEST_CHECKPOINT_API="+hfAPI, "GOINFER_TEST_CHECKPOINT_CDN="+hfCDN)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("incomplete pull: %v\n%s", err, out)
	}
	stage := testCheckpointStagePath(dest)
	before := dirBytes(stage)
	const keep = "obsolete-name.safetensors"
	const replaced = "model-00001-of-00002.safetensors"
	if err := os.WriteFile(filepath.Join(stage, keep), []byte("unrecorded"), 0o644); err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(stage, "replacement")
	if err := os.WriteFile(other, []byte("different inode at an old shard path"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(other, filepath.Join(stage, replaced)); err != nil {
		t.Fatal(err)
	}
	single, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	h.files = map[string][]byte{"config.json": files["config.json"], checkpointWeights: single, tokenizer: files[tokenizer]}
	h.mu.Unlock()
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := download(t.Context(), dest, nil); err != nil {
		t.Fatal(err)
	}
	after := dirBytes(stage)
	workspaceBytes := dirBytes(filepath.Dir(stage))
	t.Logf("stage bytes before format switch %d, after successful publication %d; total workspace after publication %d", before, after, workspaceBytes)
	if workspaceBytes >= before/2 {
		t.Error("old weight bytes remained in staging")
	}
	if raw, err := os.ReadFile(filepath.Join(stage, keep)); err != nil || string(raw) != "unrecorded" {
		t.Errorf("unrecorded file changed: %q, %v", raw, err)
	}
	if raw, err := os.ReadFile(filepath.Join(stage, replaced)); err != nil || string(raw) != "different inode at an old shard path" {
		t.Errorf("the replacement at an old shard path was not kept: %q, %v", raw, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Error("published checkpoint failed verification")
	}
}
