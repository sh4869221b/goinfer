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

func testCheckpointStagePath(dest string) string {
	return filepath.Join(filepath.Dir(dest), checkpointWorkspaceName(filepath.Base(dest)), checkpointStageName)
}

type testCheckpointPartial struct {
	path string
	data []byte
}

func (w *checkpointWorkspace) recordPartial(stage *os.Root, name string, info os.FileInfo) {
	path := filepath.ToSlash(name)
	w.recordReceipt(checkpointReceipt{Owner: w.owner, Path: path, Partial: true}, info)
	w.pruneTransferReceipts(context.Background(), stage, path, nil)
}

func createTestCheckpointStage(t *testing.T, dest string, partials ...testCheckpointPartial) string {
	t.Helper()
	parent, err := os.OpenRoot(filepath.Dir(dest))
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()
	work, err := lockCheckpointDestination(parent, filepath.Base(dest))
	if err != nil {
		t.Fatal(err)
	}
	defer work.Close()
	stage, err := work.openStage(true)
	if err != nil {
		t.Fatal(err)
	}
	defer stage.Close()
	for _, part := range partials {
		if err := stage.MkdirAll(filepath.Dir(part.path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := stage.WriteFile(part.path, part.data, 0o644); err != nil {
			t.Fatal(err)
		}
		info, err := stage.Lstat(part.path)
		if err != nil {
			t.Fatal(err)
		}
		work.recordPartial(stage, part.path, info)
		if !work.ownsPartial(part.path, info) {
			t.Fatalf("partial fixture %s was not recorded", part.path)
		}
	}
	return testCheckpointStagePath(dest)
}

func TestCheckpointPreservesLegacyWorkspaceUsedAsDestination(t *testing.T) {
	const repo = "o/tiny"
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	parent := t.TempDir()
	legacy := filepath.Join(parent, ".model.pull", "checkpoint")
	if _, err := download(t.Context(), legacy, nil); err != nil {
		t.Fatal(err)
	}
	notes := filepath.Join(legacy, "notes.txt")
	if err := os.WriteFile(notes, []byte("keep me"), 0o644); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(parent, "model")
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Errorf("independent pull = %q, %v", got, err)
	}
	if got, err := os.ReadFile(notes); err != nil || string(got) != "keep me" {
		t.Errorf("other destination lost notes: %q, %v", got, err)
	}
	if _, ok := CachedCheckpoint(legacy); !ok {
		t.Error("other destination no longer verifies")
	}
}

func TestCheckpointIsolatesStageFromPreviousWriter(t *testing.T) {
	const child = "GOINFER_TEST_PREVIOUS_CHECKPOINT_STAGE"
	const aux = "tokenizer_config.json"
	if stage := os.Getenv(child); stage != "" {
		root, err := os.OpenRoot(stage)
		if err != nil {
			t.Fatal(err)
		}
		defer root.Close()
		fmt.Println("opened")
		if _, err := bufio.NewReader(os.Stdin).ReadByte(); err != nil {
			t.Fatal(err)
		}
		if err := root.WriteFile(aux, []byte(`{"model_max_length":111}`), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	files[aux] = []byte(`{"model_max_length":222}`)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "model")
	legacy := dest + ".partial"
	if err := os.Mkdir(legacy, 0o700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointIsolatesStageFromPreviousWriter$")
	cmd.Env = append(os.Environ(), child+"="+legacy)
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
	if line, err := bufio.NewReader(out).ReadString('\n'); err != nil || strings.TrimSpace(line) != "opened" {
		t.Fatalf("previous writer did not open stage: %q, %v", line, err)
	}
	if _, err := download(t.Context(), dest, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(in, "1"); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if raw, err := os.ReadFile(filepath.Join(dest, aux)); err != nil || string(raw) != string(files[aux]) {
		t.Errorf("old writer modified publication: %q, %v", raw, err)
	}
	if raw, err := os.ReadFile(filepath.Join(legacy, aux)); err != nil || string(raw) != `{"model_max_length":111}` {
		t.Errorf("old stage was moved or modified: %q, %v", raw, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Error("old writer invalidated published cache")
	}
}

func TestCheckpointPullIgnoresInaccessibleLegacyWorkspace(t *testing.T) {
	const repo = "o/tiny"
	shared := t.TempDir()
	legacy := filepath.Join(shared, ".goinfer-pulls")
	if err := os.Mkdir(legacy, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(legacy, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(legacy, 0o700) })
	if _, err := os.Stat(filepath.Join(legacy, "other-user")); !os.IsPermission(err) {
		t.Skipf("fixture requires a legacy workspace that cannot be traversed: %v", err)
	}
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(shared, "checkpoint")
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("pull with an inaccessible legacy workspace = %q, %v", got, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Error("the published checkpoint did not verify")
	}
}

func TestCheckpointLocksAreIndependentAcrossDestinations(t *testing.T) {
	parent, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()
	for _, dest := range []string{"first", "second"} {
		work, err := lockCheckpointDestination(parent, dest)
		if err != nil {
			t.Fatal(err)
		}
		defer work.Close()
		if want := "." + dest + ".pull2"; work.name != want {
			t.Errorf("workspace for %s = %q, want %q", dest, work.name, want)
		}
		if work, err := lockCheckpointDestination(parent, dest); err == nil || !strings.Contains(err.Error(), "another checkpoint pull") {
			if work != nil {
				work.Close()
			}
			t.Errorf("second lock for %s = %v, want a lock conflict", dest, err)
		}
	}
}

func TestCheckpointParallelPullCannotWriteToPublishedDirectory(t *testing.T) {
	const repo = "o/tiny"
	const generation = "generation_config.json"
	old := []byte(`{"eos_token_id":17}`)
	files := shardedTinyRepo(t)
	files[generation] = old
	h := &fakeHF{files: files}
	h.serve(t, repo)
	ctx := t.Context()
	_, download, err := PrepareCheckpointFor(ctx, repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	files[generation] = []byte(`{"eos_token_id":23}`)
	h.mu.Unlock()
	_, competingDownload, err := PrepareCheckpointFor(ctx, repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "checkpoint")
	var attempted bool
	progress := func(_, _ int64, name string) {
		if name != "config.json" || attempted {
			return
		}
		attempted = true
		h.mu.Lock()
		requests := h.requests
		h.mu.Unlock()
		if got, err := competingDownload(ctx, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "another checkpoint pull") {
			t.Errorf("concurrent pull = %q, %v; want a lock conflict", got, err)
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.requests != requests {
			t.Error("the losing pull reached the download")
		}
	}
	if got, err := download(ctx, dest, progress); got != dest || err != nil || !attempted {
		t.Fatalf("winning pull = %q, %v; tested a concurrent pull = %t", got, err, attempted)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Error("published checkpoint did not verify")
	}
	if got, err := competingDownload(ctx, dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), generation+" in ") {
		t.Errorf("competing plan was accepted after publication: %q, %v", got, err)
	}
	if raw, err := os.ReadFile(filepath.Join(dest, generation)); err != nil || string(raw) != string(old) {
		t.Errorf("competing plan changed published generation settings: %q, %v", raw, err)
	}
}

func TestCheckpointRecoversStageAfterKilledPublisher(t *testing.T) {
	const child = "GOINFER_TEST_CLAIMED_CHECKPOINT"
	const name = "checkpoint"
	const interruptedPublish = "publish-interrupted"
	const completed = "model-00001-of-00002.safetensors"
	const interrupted = "model-00002-of-00002.safetensors"
	if dir := os.Getenv(child); dir != "" {
		parent, err := os.OpenRoot(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer parent.Close()
		work, err := lockCheckpointDestination(parent, name)
		if err != nil {
			t.Fatal(err)
		}
		defer work.Close()
		stage, err := work.openStage(false)
		if err != nil || stage == nil {
			t.Fatalf("open resumed stage: %v", err)
		}
		defer stage.Close()
		if err := work.root.Mkdir(interruptedPublish, 0o700); err != nil {
			t.Fatal(err)
		}
		publish, err := work.root.OpenRoot(interruptedPublish)
		if err != nil {
			t.Fatal(err)
		}
		defer publish.Close()
		info, err := stage.Lstat(completed)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := preparePublishedFile(t.Context(), stage, publish, File{Path: completed, Size: info.Size()}); err != nil {
			t.Fatal(err)
		}
		fmt.Println("prepared")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	const repo = "o/tiny"
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	dest := filepath.Join(dir, name)
	stagePath := createTestCheckpointStage(t, dest, testCheckpointPartial{interrupted + ".part", files[interrupted][:32]})
	if err := os.WriteFile(filepath.Join(stagePath, completed), files[completed], 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointRecoversStageAfterKilledPublisher$")
	cmd.Env = append(os.Environ(), child+"="+dir)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer stdin.Close()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		if cmd.ProcessState == nil {
			_ = cmd.Wait()
		}
	})
	if line, err := bufio.NewReader(stdout).ReadString('\n'); strings.TrimSpace(line) != "prepared" || err != nil {
		t.Fatalf("waiting for the child to stage publication: %q, %v", line, err)
	}
	if got, err := download(t.Context(), dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "another checkpoint pull") {
		t.Fatalf("live owner's claim was not protected: %q, %v", got, err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("child exited normally instead of being killed")
	}
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("retry after killing the writer = %q, %v", got, err)
	}
	claim := filepath.Join(dir, checkpointWorkspaceName(name), interruptedPublish, completed)
	if raw, err := os.ReadFile(claim); err != nil || string(raw) != string(files[completed]) {
		t.Errorf("retry modified the abandoned publication: %v", err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Error("recovered checkpoint did not verify")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.hits[completed] != 0 || h.ranges[interrupted] != "bytes=32-" {
		t.Errorf("recovery did not reuse staged weights: hits %v, ranges %v", h.hits, h.ranges)
	}
}

func TestCheckpointPreservesUnownedRecoveryDirectories(t *testing.T) {
	const repo = "o/tiny"
	for _, fixture := range []string{"unowned workspace", "unowned stage", "other destination"} {
		t.Run(fixture, func(t *testing.T) {
			h := &fakeHF{files: shardedTinyRepo(t)}
			h.serve(t, repo)
			_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
			if err != nil {
				t.Fatal(err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			stage := testCheckpointStagePath(dest)
			switch fixture {
			case "unowned stage":
				createTestCheckpointStage(t, dest)
				if err := os.Remove(filepath.Join(stage, checkpointOwnerName)); err != nil {
					t.Fatal(err)
				}
			case "other destination":
				other := filepath.Join(filepath.Dir(dest), "other")
				otherStage := createTestCheckpointStage(t, other)
				if err := os.Rename(filepath.Dir(otherStage), filepath.Dir(stage)); err != nil {
					t.Fatal(err)
				}
			default:
				if err := os.MkdirAll(stage, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			file := filepath.Join(stage, "notes.txt")
			if err := os.WriteFile(file, []byte("leave in place"), 0o644); err != nil {
				t.Fatal(err)
			}
			if got, err := download(t.Context(), dest, nil); got != "" || err == nil || !strings.Contains(err.Error(), "no verified owner") {
				t.Errorf("unowned stage pull = %q, %v", got, err)
			}
			if raw, err := os.ReadFile(file); err != nil || string(raw) != "leave in place" {
				t.Errorf("unowned stage was modified: %q, %v", raw, err)
			}
			if _, err := os.Lstat(dest); !os.IsNotExist(err) {
				t.Errorf("unowned stage was published: %v", err)
			}
		})
	}
}
