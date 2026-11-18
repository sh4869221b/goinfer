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

const (
	checkpointCrashRepo  = "o/tiny"
	checkpointCrashShard = "model-00001-of-00002.safetensors"
	checkpointCrashMode  = "GOINFER_TEST_CHECKPOINT_CRASH_MODE"
	checkpointCrashDest  = "GOINFER_TEST_CHECKPOINT_CRASH_DEST"
)

func TestCheckpointTransferCrashBoundaries(t *testing.T) {
	if mode := os.Getenv(checkpointCrashMode); mode != "" {
		hfAPI = os.Getenv("GOINFER_TEST_CHECKPOINT_API")
		hfCDN = os.Getenv("GOINFER_TEST_CHECKPOINT_CDN")
		dest := os.Getenv(checkpointCrashDest)
		pause := func() {
			fmt.Println("ready")
			_, _ = io.Copy(io.Discard, os.Stdin)
		}
		switch mode {
		case "create", "create collision":
			create := createCheckpointPartFile
			createCheckpointPartFile = func(root *os.Root, name string, flags int) (*os.File, error) {
				f, err := create(root, name, flags)
				if err == nil {
					pause()
				}
				return f, err
			}
		case "rename", "replaced":
			renameCompletedDownload = func(root *os.Root, src, dest string) error {
				if err := root.Rename(src, dest); err != nil {
					return err
				}
				if filepath.Base(dest) == checkpointCrashShard {
					pause()
				}
				return nil
			}
		}
		_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = download(t.Context(), dest, nil)
		if mode == "other version" && err != nil && strings.Contains(err.Error(), checkpointCrashShard) {
			pause()
			return
		}
		t.Fatalf("pull returned before the crash boundary: %v", err)
	}

	for _, mode := range []string{"other version", "create", "create collision", "rename", "replaced"} {
		t.Run(mode, func(t *testing.T) {
			split := shardedTinyRepo(t)
			weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			single := map[string][]byte{"config.json": split["config.json"], checkpointWeights: weights}
			h := &fakeHF{files: split}
			h.serve(t, checkpointCrashRepo)
			parent := t.TempDir()
			dest := filepath.Join(parent, "checkpoint")
			backup := filepath.Join(parent, "backup")
			stage := testCheckpointStagePath(dest)
			part := filepath.Join(stage, checkpointCrashShard+".part")
			final := filepath.Join(stage, checkpointCrashShard)
			if mode == "other version" {
				setCrashRepoFiles(h, single, "")
				pullCrashCheckpoint(t, dest)
				if err := os.Rename(dest, backup); err != nil {
					t.Fatal(err)
				}
				setCrashRepoFiles(h, split, checkpointCrashShard)
			}
			killCheckpointAtTransferBoundary(t, mode, dest)
			if mode == "other version" {
				before, err := os.ReadFile(part)
				if err != nil || len(before) == 0 {
					t.Fatalf("interrupted version B has no staged prefix: %v", err)
				}
				if err := os.Rename(backup, dest); err != nil {
					t.Fatal(err)
				}
				setCrashRepoFiles(h, single, "")
				pullCrashCheckpoint(t, dest)
				if after, err := os.ReadFile(part); err != nil || !bytes.Equal(after, before) {
					t.Fatalf("repull of version A changed version B's prefix: %v", err)
				}
				if err := os.Rename(dest, backup); err != nil {
					t.Fatal(err)
				}
				setCrashRepoFiles(h, split, "")
				pullCrashCheckpoint(t, dest)
				h.mu.Lock()
				defer h.mu.Unlock()
				if got, want := h.ranges[checkpointCrashShard], fmt.Sprintf("bytes=%d-", len(before)); got != want {
					t.Errorf("version B Range = %q, want %q", got, want)
				}
				return
			}
			if mode == "create" {
				pullCrashCheckpoint(t, dest)
				return
			}
			if mode == "create collision" {
				// A third party can create an empty canonical partial during the interruption.
				unknown, err := os.OpenFile(part, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
				if err != nil {
					t.Fatal(err)
				}
				original, err := unknown.Stat()
				unknown.Close()
				if err != nil {
					t.Fatal(err)
				}
				_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := download(t.Context(), dest, nil); err == nil || !strings.Contains(err.Error(), "no verified transfer record") {
					t.Errorf("expected refusal of the third-party empty file, got %v", err)
				}
				if info, err := os.Stat(part); err != nil || info.Size() != 0 || !os.SameFile(info, original) {
					t.Fatalf("third-party empty file changed: %v", err)
				}
				if err := os.Remove(part); err != nil {
					t.Fatal(err)
				}
				pullCrashCheckpoint(t, dest)
				return
			}
			info, err := os.Stat(final)
			if err != nil || info.Size() != int64(len(split[checkpointCrashShard])) {
				t.Fatalf("crash did not leave a completed shard in staging: %v", err)
			}
			replacement := bytes.Clone(split[checkpointCrashShard])
			replacement[0] ^= 1
			if mode == "replaced" {
				if err := os.WriteFile(final+".new", replacement, 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(final+".new", info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(final+".new", final); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(part, []byte("keep"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			setCrashRepoFiles(h, single, "")
			pullCrashCheckpoint(t, dest)
			if mode == "replaced" {
				if raw, err := os.ReadFile(final); err != nil || !bytes.Equal(raw, replacement) {
					t.Errorf("replacement inode was not preserved: %v", err)
				}
				if raw, err := os.ReadFile(part); err != nil || string(raw) != "keep" {
					t.Errorf("unrelated .part was not preserved: %q, %v", raw, err)
				}
			} else if _, err := os.Lstat(final); !os.IsNotExist(err) {
				t.Errorf("%d obsolete bytes remained after format switch: %v", info.Size(), err)
			}
		})
	}
}

func TestCheckpointPullWhenPartialLinkUnavailable(t *testing.T) {
	link := linkCheckpointPartial
	linkCheckpointPartial = func(*os.Root, string, string) error { return os.ErrPermission }
	t.Cleanup(func() { linkCheckpointPartial = link })
	h := &fakeHF{files: shardedTinyRepo(t), failOnce: checkpointCrashShard}
	h.serve(t, checkpointCrashRepo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := download(t.Context(), dest, nil); err == nil || !strings.Contains(err.Error(), checkpointCrashShard) {
		t.Fatalf("expected interrupted transfer, got %v", err)
	}
	stage := testCheckpointStagePath(dest)
	if _, err := os.Lstat(filepath.Join(stage, checkpointCrashShard+".part")); !os.IsNotExist(err) {
		t.Errorf("private transfer exposed a .part without a hard link: %v", err)
	}
	pullCrashCheckpoint(t, dest)
	h.mu.Lock()
	defer h.mu.Unlock()
	if got := h.ranges[checkpointCrashShard]; got != "" {
		t.Errorf("private transfer restarted with Range %q", got)
	}
	if receipts, err := filepath.Glob(filepath.Join(filepath.Dir(stage), checkpointReceiptPrefix+"*")); err != nil || len(receipts) != 0 {
		t.Errorf("private transfer records remain: %v, %v", receipts, err)
	}
}

func killCheckpointAtTransferBoundary(t *testing.T, mode, dest string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckpointTransferCrashBoundaries$")
	cmd.Env = append(os.Environ(), checkpointCrashMode+"="+mode, checkpointCrashDest+"="+dest, "GOINFER_TEST_CHECKPOINT_API="+hfAPI, "GOINFER_TEST_CHECKPOINT_CDN="+hfCDN)
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
	if line, err := bufio.NewReader(out).ReadString('\n'); line != "ready\n" || err != nil {
		t.Fatalf("process did not reach %s: %q, %v", mode, line, err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("writer exited normally")
	}
}

func setCrashRepoFiles(h *fakeHF, files map[string][]byte, fail string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.files, h.failOnce = files, fail
}

func pullCrashCheckpoint(t *testing.T, dest string) {
	t.Helper()
	_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("pull = %q, %v", got, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Fatal("published checkpoint failed offline verification")
	}
}
