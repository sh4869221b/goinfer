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
		lock, claim, err := lockCheckpointDestination(parent, dest)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		if want := filepath.Join("."+dest+".pull", "checkpoint"); claim != want {
			t.Errorf("claim for %s = %q, want %q", dest, claim, want)
		}
		if lock, _, err := lockCheckpointDestination(parent, dest); err == nil || !strings.Contains(err.Error(), "another checkpoint pull") {
			if lock != nil {
				lock.Close()
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

func TestCheckpointRecoversClaimAfterKilledProcess(t *testing.T) {
	const child = "GOINFER_TEST_CLAIMED_CHECKPOINT"
	const name = "checkpoint"
	const staging = name + ".partial"
	if dir := os.Getenv(child); dir != "" {
		parent, err := os.OpenRoot(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer parent.Close()
		lock, claim, err := lockCheckpointDestination(parent, name)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		if err := parent.Rename(staging, claim); err != nil {
			t.Fatal(err)
		}
		fmt.Println("claimed")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	const repo = "o/tiny"
	const completed = "model-00001-of-00002.safetensors"
	const interrupted = "model-00002-of-00002.safetensors"
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	stagePath, dest := filepath.Join(dir, staging), filepath.Join(dir, name)
	if err := os.Mkdir(stagePath, 0o755); err != nil {
		t.Fatal(err)
	}
	for filename, raw := range map[string][]byte{completed: files[completed], interrupted + ".part": files[interrupted][:32]} {
		if err := os.WriteFile(filepath.Join(stagePath, filename), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointRecoversClaimAfterKilledProcess$")
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
	if line, err := bufio.NewReader(stdout).ReadString('\n'); strings.TrimSpace(line) != "claimed" || err != nil {
		t.Fatalf("waiting for the child to claim staging: %q, %v", line, err)
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
	claim := filepath.Join(dir, checkpointWorkspaceName(name), "checkpoint")
	if _, err := os.Lstat(claim); !os.IsNotExist(err) {
		t.Errorf("publish claim still present after retry: %v", err)
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

func TestCheckpointPreservesConflictingRecoveryDirectories(t *testing.T) {
	const dest = "checkpoint"
	parent, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()
	lock, claim, err := lockCheckpointDestination(parent, dest)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	paths := map[string]string{dest + ".partial": "normal stage", claim: "interrupted publish"}
	for dir, data := range paths {
		if err := parent.Mkdir(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := parent.WriteFile(filepath.Join(dir, "data"), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := recoverCheckpointStage(parent, dest+".partial", claim); err == nil || !strings.Contains(err.Error(), "contain staging data") {
		t.Errorf("expected a recovery conflict, got %v", err)
	}
	for dir, want := range paths {
		if got, err := parent.ReadFile(filepath.Join(dir, "data")); err != nil || string(got) != want {
			t.Errorf("conflict changed %s: %q, %v", dir, got, err)
		}
	}
}
