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
	checkpointCrashTail  = "model-00002-of-00002.safetensors"
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
		case "rename", "replaced", "edited", "edited same stat":
			renameCompletedDownload = func(root *os.Root, src, dest string) error {
				if err := root.Rename(src, dest); err != nil {
					return err
				}
				if filepath.Base(dest) == checkpointCrashShard {
					pause()
				}
				return nil
			}
		case "published":
			renameCheckpointPublication = func(root *os.Root, src, dest string) error {
				if err := root.Rename(src, dest); err != nil {
					return err
				}
				pause()
				return nil
			}
		}
		_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = download(t.Context(), dest, func(_, _ int64, name string) {
			if mode == "after shard" && name == checkpointCrashShard {
				path := filepath.Join(testCheckpointStagePath(dest), name)
				if _, err := os.Stat(path); err == nil {
					pause()
				}
			}
		})
		if (mode == "other version" || mode == "older writer") && err != nil && strings.Contains(err.Error(), checkpointCrashShard) {
			pause()
			return
		}
		if mode == "older reuse" && err != nil && strings.Contains(err.Error(), checkpointCrashTail) {
			pause()
			return
		}
		t.Fatalf("pull returned before the crash boundary: %v", err)
	}

	for _, mode := range []string{"other version", "older writer", "create", "create collision", "rename", "replaced", "edited", "edited same stat"} {
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
			legacy := os.Getenv("GOINFER_TEST_LEGACY_PULL_BINARY")
			if mode == "older writer" && legacy != "" {
				shared := filepath.Join(parent, "shared")
				killCheckpointAtTransferBoundary(t, "older lock", shared, func() {
					_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := download(t.Context(), shared, nil); err == nil || !strings.Contains(err.Error(), "another checkpoint pull") {
						t.Fatalf("current writer was not excluded by old writer: %v", err)
					}
					t.Log("current writer was excluded by the legacy process")
				})
			}
			var oldPublication []byte
			if mode == "other version" || mode == "older writer" {
				setCrashRepoFiles(h, single, "")
				pullCrashCheckpoint(t, dest)
				if err := os.Rename(dest, backup); err != nil {
					t.Fatal(err)
				}
				if mode == "older writer" {
					oldPublication, err = os.ReadFile(filepath.Join(stage, checkpointPublicationName))
					if err != nil {
						t.Fatal(err)
					}
				}
				setCrashRepoFiles(h, split, checkpointCrashShard)
			}
			var contention func()
			if mode == "create" && legacy != "" {
				contention = func() {
					cmd := checkpointCrashCommand(t.Context(), legacy, "legacy blocked", dest)
					if out, err := cmd.CombinedOutput(); err != nil || !strings.Contains(string(out), "locked") {
						t.Fatalf("legacy writer was not excluded by current writer: %v\n%s", err, out)
					}
					t.Log("legacy writer was excluded by the current process")
				}
			}
			killCheckpointAtTransferBoundary(t, mode, dest, contention)
			if mode == "older writer" && legacy == "" {
				// Older writers use the same lock and receipts but leave the publication record alone.
				if err := os.WriteFile(filepath.Join(stage, checkpointPublicationName), oldPublication, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "other version" || mode == "older writer" {
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
			if mode == "edited" || mode == "edited same stat" {
				if mode == "edited" {
					replacement = []byte("local replacement written in place")
				}
				if err := os.WriteFile(final, replacement, 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(final, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				if after, err := os.Stat(final); err != nil || !os.SameFile(after, info) {
					t.Fatalf("edit did not preserve inode: %v", err)
				}
			}
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
			} else if mode == "edited" || mode == "edited same stat" {
				if raw, err := os.ReadFile(final); err != nil || !bytes.Equal(raw, replacement) {
					t.Errorf("edit of completed inode was not preserved: %v", err)
				}
			} else if _, err := os.Lstat(final); !os.IsNotExist(err) {
				t.Errorf("%d obsolete bytes remained after format switch: %v", info.Size(), err)
			}
		})
	}
}

func TestCheckpointRechecksCompletedStageBytes(t *testing.T) {
	for _, changeLayout := range []bool{false, true} {
		t.Run(fmt.Sprintf("layout switch=%t", changeLayout), func(t *testing.T) {
			files := shardedTinyRepo(t)
			h := &fakeHF{files: files}
			h.serve(t, checkpointCrashRepo)
			dest := filepath.Join(t.TempDir(), "checkpoint")
			killCheckpointAtTransferBoundary(t, "after shard", dest, nil)
			stageFile := filepath.Join(testCheckpointStagePath(dest), checkpointCrashShard)
			info, err := os.Stat(stageFile)
			if err != nil {
				t.Fatal(err)
			}
			edited := bytes.Clone(files[checkpointCrashShard])
			edited[len(edited)-1] ^= 1
			if err := os.WriteFile(stageFile, edited, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.Chtimes(stageFile, info.ModTime(), info.ModTime()); err != nil {
				t.Fatal(err)
			}
			if after, err := os.Stat(stageFile); err != nil || !sameCheckpointFileState(info, after) {
				t.Fatalf("failed to preserve inode, size and mtime: %v", err)
			}
			if changeLayout {
				weights, err := os.ReadFile("../testdata/llama-tiny/model.safetensors")
				if err != nil {
					t.Fatal(err)
				}
				setCrashRepoFiles(h, map[string][]byte{"config.json": files["config.json"], checkpointWeights: weights}, "")
				pullCrashCheckpoint(t, dest)
				if raw, err := os.ReadFile(stageFile); err != nil || !bytes.Equal(raw, edited) {
					t.Fatalf("cleanup discarded edited bytes: %v", err)
				}
				return
			}
			_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := download(t.Context(), dest, nil); err != nil {
				if _, ok := CachedCheckpoint(dest); ok || !strings.Contains(err.Error(), "digest") {
					t.Fatalf("unexpected result for edited shard: cache=%t, err=%v", ok, err)
				}
				pullCrashCheckpoint(t, dest)
			}
			if raw, err := os.ReadFile(filepath.Join(dest, checkpointCrashShard)); err != nil || !bytes.Equal(raw, files[checkpointCrashShard]) {
				_, cached := CachedCheckpoint(dest)
				t.Fatalf("published wrong bytes with CachedCheckpoint=%t: %v", cached, err)
			}
		})
	}
}

func TestCheckpointDefersCleanupForLegacyWriterReuse(t *testing.T) {
	a := shardedTinyRepo(t)
	b := shardedTinyRepo(t)
	b[checkpointCrashTail][len(b[checkpointCrashTail])-1] ^= 1
	h := &fakeHF{files: a}
	h.serve(t, checkpointCrashRepo)
	parent := t.TempDir()
	dest, backup := filepath.Join(parent, "checkpoint"), filepath.Join(parent, "backup")
	killCheckpointAtTransferBoundary(t, "published", dest, nil)
	if err := os.Rename(dest, backup); err != nil {
		t.Fatal(err)
	}
	stage := testCheckpointStagePath(dest)
	marker := filepath.Join(stage, checkpointPublicationName)
	publication, err := os.ReadFile(marker)
	if err != nil {
		t.Fatal(err)
	}
	setCrashRepoFiles(h, b, checkpointCrashTail)
	killCheckpointAtTransferBoundary(t, "older reuse", dest, nil)
	if os.Getenv("GOINFER_TEST_LEGACY_PULL_BINARY") == "" {
		// The older executable leaves the publication marker untouched when it starts a pull.
		if err := os.WriteFile(marker, publication, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	part := filepath.Join(stage, checkpointCrashTail+".part")
	before, err := os.ReadFile(part)
	if err != nil || len(before) == 0 {
		t.Fatalf("legacy B has no resumable prefix: %v", err)
	}
	h.mu.Lock()
	hits := h.hits[checkpointCrashShard]
	h.mu.Unlock()
	if hits != 1 {
		t.Fatalf("legacy B fetched the shared shard instead of reusing it: %d requests", hits)
	}
	if err := os.Rename(backup, dest); err != nil {
		t.Fatal(err)
	}
	setCrashRepoFiles(h, a, "")
	pullCrashCheckpoint(t, dest)
	if raw, err := os.ReadFile(filepath.Join(stage, checkpointCrashShard)); err != nil || !bytes.Equal(raw, b[checkpointCrashShard]) {
		t.Fatalf("retry of A removed the shard B reused: %v", err)
	}
	if raw, err := os.ReadFile(part); err != nil || !bytes.Equal(raw, before) {
		t.Fatalf("retry of A changed the partial from B: %v", err)
	}
	if err := os.Rename(dest, backup); err != nil {
		t.Fatal(err)
	}
	setCrashRepoFiles(h, b, "")
	pullCrashCheckpoint(t, dest)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.hits[checkpointCrashShard] != hits || h.ranges[checkpointCrashTail] != fmt.Sprintf("bytes=%d-", len(before)) {
		t.Errorf("B did not resume the shared shard and tail: hits=%v, ranges=%v", h.hits, h.ranges)
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

func TestCheckpointCacheEntriesIgnorePublicationRecord(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, checkpointCrashRepo)
	dest, err := CacheDir(checkpointCrashRepo)
	if err != nil {
		t.Fatal(err)
	}
	pullCrashCheckpoint(t, dest)
	if err := os.Rename(dest, filepath.Join(t.TempDir(), "completed")); err != nil {
		t.Fatal(err)
	}
	entries, err := CacheEntries()
	if err != nil || len(entries) != 0 {
		t.Fatalf("cache contains a record-only checkpoint: %+v, %v", entries, err)
	}
	notes := []byte("local file should count")
	if err := os.WriteFile(filepath.Join(testCheckpointStagePath(dest), "notes.txt"), notes, 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err = CacheEntries()
	if err != nil || len(entries) != 1 || entries[0].Bytes != int64(len(notes)) || entries[0].Complete {
		t.Errorf("cache should count only the staged data: %+v, %v", entries, err)
	}
}

func killCheckpointAtTransferBoundary(t *testing.T, mode, dest string, during func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	var legacy string
	if mode == "older writer" || mode == "older lock" || mode == "older reuse" {
		legacy = os.Getenv("GOINFER_TEST_LEGACY_PULL_BINARY")
	}
	cmd := checkpointCrashCommand(ctx, legacy, mode, dest)
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
	if during != nil {
		during()
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("writer exited normally")
	}
}

func checkpointCrashCommand(ctx context.Context, legacy, mode, dest string) *exec.Cmd {
	binary, test := os.Args[0], "-test.run=^TestCheckpointTransferCrashBoundaries$"
	if legacy != "" {
		binary, test = legacy, "-test.run=^TestLegacyCheckpointWriter$"
	}
	cmd := exec.CommandContext(ctx, binary, test)
	cmd.Env = append(os.Environ(), checkpointCrashMode+"="+mode, checkpointCrashDest+"="+dest, "GOINFER_TEST_CHECKPOINT_API="+hfAPI, "GOINFER_TEST_CHECKPOINT_CDN="+hfCDN)
	return cmd
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
