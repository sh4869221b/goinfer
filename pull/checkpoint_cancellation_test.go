package pull

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCheckpointCancelAfterRenameWithoutLinks(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("requires a filesystem change time")
	}
	for _, mode := range []string{"unchanged", "edited", "replaced", "symlink", "legacy cleanup record", "missing receipt", "partly reclaimed", "unrecorded file replaced"} {
		t.Run(mode, func(t *testing.T) {
			files := shardedTinyRepo(t)
			h := &fakeHF{files: files}
			h.serve(t, checkpointCrashRepo)
			_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
			if err != nil {
				t.Fatal(err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			link, rename := linkCheckpointTransfer, renameCheckpointPublication
			t.Cleanup(func() { linkCheckpointTransfer, renameCheckpointPublication = link, rename })
			linkCheckpointTransfer = func(*os.Root, string, string) error { return errors.ErrUnsupported }
			renameCheckpointPublication = func(ctx context.Context, root *os.Root, src, dest string) error {
				err := rename(ctx, root, src, dest)
				cancel()
				return err
			}
			stage := testCheckpointStagePath(dest)
			note := filepath.Join(stage, "user-notes.txt")
			if got, err := download(ctx, dest, func(done, total int64, _ string) {
				if done == total {
					if err := os.WriteFile(note, []byte("keep"), 0o644); err != nil {
						t.Fatal(err)
					}
				}
			}); got != dest || err != nil || ctx.Err() != context.Canceled {
				t.Fatalf("canceled after rename without links = %q, %v, context=%v", got, err, ctx.Err())
			}
			renameCheckpointPublication = rename
			staged := filepath.Join(stage, checkpointCrashShard)
			info, err := os.Stat(staged)
			if err != nil {
				t.Fatalf("cancellation did not defer stage cleanup: %v", err)
			}
			want := files[checkpointCrashShard]
			if mode == "legacy cleanup record" {
				root, err := os.OpenRoot(stage)
				if err != nil {
					t.Fatal(err)
				}
				defer root.Close()
				record, err := readCheckpointCleanup(root)
				if err != nil {
					t.Fatal(err)
				}
				record.StageInventory = nil
				raw, err := json.Marshal(struct {
					checkpointCleanup
					StageEntries []string `json:"stage_entries"`
				}{record, []string{checkpointCrashShard, checkpointCrashTail}})
				if err != nil {
					t.Fatal(err)
				}
				if err := replaceRootFile(root, checkpointCleanupName, raw); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "edited" {
				info, want = rewriteCheckpointWeightInPlace(t, staged, want)
			}
			if mode == "replaced" || mode == "symlink" {
				replacement := filepath.Join(t.TempDir(), "replacement")
				if err := os.WriteFile(replacement, want, 0o644); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(replacement, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
				if mode == "replaced" {
					if err := os.Rename(replacement, staged); err != nil {
						t.Fatal(err)
					}
				} else {
					if err := os.Remove(staged); err != nil {
						t.Fatal(err)
					}
					saveCheckpointConfigLink(t, replacement, staged)
				}
				info, err = os.Lstat(staged)
				if err != nil {
					t.Fatal(err)
				}
			}
			if mode == "missing receipt" || mode == "partly reclaimed" {
				receipt := testSealedCheckpointReceipt(t, stage, checkpointCrashShard)
				if mode == "partly reclaimed" {
					for _, path := range []string{staged, sidecarPath(staged)} {
						if err := os.Remove(path); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := os.RemoveAll(receipt); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "unrecorded file replaced" {
				replacement := filepath.Join(t.TempDir(), "replacement")
				original, err := os.Stat(note)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(replacement, []byte("keep"), 0o644); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(replacement, original.ModTime(), original.ModTime()); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(replacement, note); err != nil {
					t.Fatal(err)
				}
			}
			if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
				t.Fatalf("retry of committed checkpoint without links = %q, %v", got, err)
			}
			if mode == "unchanged" || mode == "partly reclaimed" {
				for _, shard := range []string{checkpointCrashShard, checkpointCrashTail} {
					path := filepath.Join(stage, shard)
					for _, name := range []string{path, sidecarPath(path)} {
						if _, err := os.Lstat(name); !os.IsNotExist(err) {
							t.Errorf("retry retained staged file %s: %v", name, err)
						}
					}
				}
			} else {
				if raw, err := os.ReadFile(staged); err != nil || !bytes.Equal(raw, want) {
					t.Errorf("retry changed the %s stage file: %v", mode, err)
				}
				if after, err := os.Lstat(staged); err != nil || !os.SameFile(info, after) {
					t.Errorf("retry did not retain the %s staged inode: %v", mode, err)
				}
				if _, err := os.Stat(filepath.Join(stage, checkpointCrashTail)); err != nil {
					t.Errorf("retry reclaimed a shared shard in mode %s: %v", mode, err)
				}
			}
			if raw, err := os.ReadFile(note); err != nil || string(raw) != "keep" {
				t.Errorf("retry changed an unrecorded file: %q, %v", raw, err)
			}
			if _, ok := CachedCheckpoint(dest); !ok {
				t.Error("published checkpoint failed verification after cleanup")
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.hits[checkpointCrashShard] != 1 || h.hits[checkpointCrashTail] != 1 {
				t.Errorf("retry downloaded weights again: %v", h.hits)
			}
		})
	}
}

func TestCheckpointCleanupPreservesReplacedStageSidecar(t *testing.T) {
	for _, tc := range []struct {
		name          string
		recover       bool
		noLinks       bool
		unchangedJSON bool
		beforeSeal    bool
		kill          bool
	}{
		{name: "before sealing", beforeSeal: true},
		{name: "during publication"},
		{name: "after publication", recover: true},
		{name: "after SIGKILL", recover: true, kill: true},
		{name: "after publication without links", recover: true, noLinks: true},
		{name: "byte identical replacement", recover: true, unchangedJSON: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.noLinks && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
				t.Skip("requires a filesystem change time")
			}
			h := &fakeHF{files: shardedTinyRepo(t)}
			h.serve(t, checkpointCrashRepo)
			dest := filepath.Join(t.TempDir(), "checkpoint")
			stage := testCheckpointStagePath(dest)
			staged := filepath.Join(stage, checkpointCrashShard)
			sidecar := sidecarPath(staged)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			link, rename := linkCheckpointTransfer, renameCheckpointPublication
			t.Cleanup(func() { linkCheckpointTransfer, renameCheckpointPublication = link, rename })
			if tc.noLinks {
				linkCheckpointTransfer = func(*os.Root, string, string) error { return errors.ErrUnsupported }
			}
			var want []byte
			var replacement os.FileInfo
			replace := func() error {
				var err error
				want, err = os.ReadFile(sidecar)
				if err != nil {
					return err
				}
				if !tc.unchangedJSON {
					want = append(bytes.TrimSuffix(want, []byte("}")), []byte(`,"local_note":"keep"}`)...)
				}
				path := filepath.Join(t.TempDir(), "replacement")
				if err := os.WriteFile(path, want, 0o600); err != nil {
					return err
				}
				replacement, err = statCheckpointFixture(path)
				if err != nil {
					return err
				}
				return os.Rename(path, sidecar)
			}
			renameCheckpointPublication = func(ctx context.Context, root *os.Root, src, dest string) error {
				if err := rename(ctx, root, src, dest); err != nil {
					return err
				}
				if !tc.beforeSeal {
					if err := replace(); err != nil {
						return err
					}
				}
				if tc.recover {
					cancel()
				}
				return nil
			}
			if tc.kill {
				killCheckpointAtTransferBoundary(t, "published", dest, nil)
				if err := replace(); err != nil {
					t.Fatal(err)
				}
			} else {
				_, download, err := PrepareCheckpointFor(ctx, checkpointCrashRepo, nil)
				if err != nil {
					t.Fatal(err)
				}
				if got, err := download(ctx, dest, func(done, total int64, _ string) {
					if tc.beforeSeal && done == total && replacement == nil {
						if err := replace(); err != nil {
							t.Fatal(err)
						}
					}
				}); err != nil || got != dest {
					t.Fatalf("publication returned %q, %v", got, err)
				}
			}
			if tc.recover {
				renameCheckpointPublication = rename
				pullCrashCheckpoint(t, dest)
			}
			if _, err := os.Lstat(staged); !os.IsNotExist(err) {
				t.Errorf("cleanup retained the unchanged weight: %v", err)
			}
			after, err := os.Stat(sidecar)
			if err != nil || !os.SameFile(after, replacement) {
				t.Fatalf("cleanup did not preserve the replacement sidecar inode: %v", err)
			}
			if raw, err := os.ReadFile(sidecar); err != nil || !bytes.Equal(raw, want) {
				t.Fatalf("cleanup changed the replacement sidecar: %v", err)
			}
		})
	}
}

func TestResolveCanceledCheckpointWithReadOnlyCache(t *testing.T) {
	withCacheRoot(t)
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, checkpointCrashRepo)
	dest, err := CacheDir(checkpointCrashRepo)
	if err != nil {
		t.Fatal(err)
	}
	pullCrashCheckpoint(t, dest)
	if err := os.Chmod(dest, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dest, 0o755) })
	h.mu.Lock()
	requests := h.requests
	h.mu.Unlock()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	ref := "hf:" + checkpointCrashRepo + ":safetensors"
	if got, err := Resolve(ctx, ref, nil); got != "" || !errors.Is(err, context.Canceled) {
		t.Errorf("canceled resolve of a read-only checkpoint = %q, %v", got, err)
	}
	if got, err := Resolve(t.Context(), ref, nil); got != dest || err != nil {
		t.Errorf("uncanceled resolve of a read-only checkpoint = %q, %v", got, err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.requests != requests {
		t.Errorf("offline resolves made %d network requests", h.requests-requests)
	}
}

func TestCheckpointCancelFromFinalProgress(t *testing.T) {
	h := &fakeHF{files: shardedTinyRepo(t)}
	h.serve(t, checkpointCrashRepo)
	plan, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "checkpoint")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	finalProgress := false
	got, err := download(ctx, dest, func(done, total int64, _ string) {
		if done == plan.Bytes && done == total {
			finalProgress = true
			cancel()
		}
	})
	if !finalProgress || got != "" || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled final progress = path %q, error %v, callback %t", got, err, finalProgress)
	}
	assertCheckpointCanceledAndResumable(t, h, dest, plan, download)
}

func TestCheckpointCancelAtPublicationRename(t *testing.T) {
	for _, after := range []bool{false, true} {
		t.Run(fmt.Sprintf("after rename=%t", after), func(t *testing.T) {
			h := &fakeHF{files: shardedTinyRepo(t)}
			h.serve(t, checkpointCrashRepo)
			plan, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
			if err != nil {
				t.Fatal(err)
			}
			dest := filepath.Join(t.TempDir(), "checkpoint")
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			rename := renameCheckpointPublication
			t.Cleanup(func() { renameCheckpointPublication = rename })
			reached := false
			renameCheckpointPublication = func(ctx context.Context, root *os.Root, src, dest string) error {
				reached = true
				if !after {
					cancel()
					return rename(ctx, root, src, dest)
				}
				err := rename(ctx, root, src, dest)
				cancel()
				return err
			}
			got, err := download(ctx, dest, nil)
			renameCheckpointPublication = rename
			if !reached {
				t.Fatalf("pull did not reach the rename boundary: %v", err)
			}
			if !after {
				if got != "" || !errors.Is(err, context.Canceled) {
					t.Fatalf("canceled before rename = %q, %v", got, err)
				}
				assertCheckpointCanceledAndResumable(t, h, dest, plan, download)
				return
			}
			if got != dest || err != nil {
				t.Fatalf("canceled after rename = %q, %v", got, err)
			}
			if _, ok := CachedCheckpoint(dest); !ok {
				t.Fatal("committed checkpoint was not usable")
			}
			staged := filepath.Join(testCheckpointStagePath(dest), checkpointCrashShard)
			if _, err := os.Stat(staged); err != nil {
				t.Fatalf("cancellation did not defer stage cleanup: %v", err)
			}
			if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
				t.Fatalf("retry of committed checkpoint = %q, %v", got, err)
			}
			if _, err := os.Lstat(staged); !os.IsNotExist(err) {
				t.Errorf("retry did not finish stage cleanup: %v", err)
			}
		})
	}
}

func TestCheckpointPublicationCancellationStopsLocalIO(t *testing.T) {
	const size = 2 * checkpointCopyChunk
	for _, mode := range []string{"copy", "hash"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			src := &checkpointCancelReader{remaining: size, cancelAfter: checkpointCopyChunk, cancel: cancel}
			var err error
			if mode == "copy" {
				err = copyCheckpointFile(ctx, io.Discard, src, size)
			} else {
				_, err = hashCheckpointFile(ctx, File{Path: checkpointCrashShard, Size: size}, src)
			}
			if !errors.Is(err, context.Canceled) || src.remaining != checkpointCopyChunk {
				t.Errorf("%s read %d bytes before returning %v, want %d bytes and cancellation", mode, size-src.remaining, err, checkpointCopyChunk)
			}
		})
	}
}

type checkpointCancelReader struct {
	remaining, cancelAfter int
	cancel                 context.CancelFunc
}

func (r *checkpointCancelReader) Read(p []byte) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	n := min(len(p), r.remaining)
	clear(p[:n])
	r.remaining -= n
	r.cancelAfter -= n
	if r.cancelAfter <= 0 {
		r.cancel()
	}
	return n, nil
}

func assertCheckpointCanceledAndResumable(t *testing.T, h *fakeHF, dest string, plan Plan, download CheckpointDownload) {
	t.Helper()
	if _, err := os.Lstat(dest); !os.IsNotExist(err) {
		t.Errorf("canceled pull published a checkpoint: %v", err)
	}
	stage := testCheckpointStagePath(dest)
	if matches, err := filepath.Glob(filepath.Join(filepath.Dir(stage), "publish-*")); err != nil || len(matches) != 0 {
		t.Errorf("canceled pull left a publication candidate: %v, %v", matches, err)
	}
	for _, file := range plan.Files {
		if info, err := os.Stat(filepath.Join(stage, filepath.FromSlash(file.Path))); err != nil || info.Size() != file.Size {
			t.Errorf("canceled pull did not retain staged %s: %v", file.Path, err)
		}
	}
	if got, err := download(t.Context(), dest, nil); got != dest || err != nil {
		t.Fatalf("retry = %q, %v", got, err)
	}
	if _, ok := CachedCheckpoint(dest); !ok {
		t.Fatal("retried checkpoint failed verification")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, shard := range []string{checkpointCrashShard, checkpointCrashTail} {
		if h.hits[shard] != 1 {
			t.Errorf("retry fetched staged shard %s again: %d requests", shard, h.hits[shard])
		}
	}
}
