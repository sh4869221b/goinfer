package pull

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

const (
	checkpointCrashRepo     = "o/tiny"
	checkpointCrashShard    = "model-00001-of-00002.safetensors"
	checkpointCrashTail     = "model-00002-of-00002.safetensors"
	checkpointCrashTemplate = "chat_template.jinja"
	checkpointCrashMode     = "GOINFER_TEST_CHECKPOINT_CRASH_MODE"
	checkpointCrashDest     = "GOINFER_TEST_CHECKPOINT_CRASH_DEST"
)

func TestCheckpointTransferCrashBoundaries(t *testing.T) {
	if mode := os.Getenv(checkpointCrashMode); mode != "" {
		hfAPI = os.Getenv("GOINFER_TEST_CHECKPOINT_API")
		hfCDN = os.Getenv("GOINFER_TEST_CHECKPOINT_CDN")
		dest := os.Getenv(checkpointCrashDest)
		if mode == "valid readonly cache" {
			if got, ok := CachedCheckpoint(dest); !ok || got != dest {
				t.Fatalf("rejected the intact read-only cache: %q, %t", got, ok)
			}
			return
		}
		if mode == "damaged cache" {
			if _, ok := CachedCheckpoint(dest); ok {
				t.Error("cached checkpoint accepted a rejected digest")
			}
			if got, err := Resolve(t.Context(), "hf:"+checkpointCrashRepo+":safetensors", nil); got != "" || err == nil {
				t.Errorf("Resolve accepted a rejected digest: %q, %v", got, err)
			}
			return
		}
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
		case "published", "published without links":
			if mode == "published without links" {
				linkCheckpointTransfer = func(*os.Root, string, string) error { return errors.ErrUnsupported }
			}
			rename := renameCheckpointPublication
			renameCheckpointPublication = func(ctx context.Context, root *os.Root, src, dest string) error {
				if err := rename(ctx, root, src, dest); err != nil {
					return err
				}
				pause()
				return nil
			}
		case "before publish":
			rename := renameCheckpointPublication
			renameCheckpointPublication = func(ctx context.Context, root *os.Root, src, dest string) error {
				pause()
				return rename(ctx, root, src, dest)
			}
		}
		_, download, err := PrepareCheckpointFor(t.Context(), checkpointCrashRepo, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := download(t.Context(), dest, func(_, _ int64, name string) {
			if mode == "after shard" && name == checkpointCrashShard || mode == "after template" && name == checkpointCrashTemplate {
				path := filepath.Join(testCheckpointStagePath(dest), name)
				if _, err := os.Stat(path); err == nil {
					pause()
				}
			}
		})
		if mode == "before publish" && err == nil && got == dest {
			return
		}
		if mode == "explicit repull" || mode == "readonly repull" {
			if got != "" || err == nil || !strings.Contains(err.Error(), "digest") {
				t.Fatalf("explicit repull accepted modified weight: %q, %v", got, err)
			}
			if mode == "readonly repull" && !errors.Is(err, os.ErrPermission) {
				t.Fatalf("expected a denied sidecar removal in the read-only cache: %v", err)
			}
			return
		}
		if mode == "after template" {
			if got != "" || err == nil || !strings.Contains(err.Error(), "digest") || !strings.Contains(err.Error(), checkpointCrashTemplate) {
				t.Fatalf("publication accepted a modified template: %q, %v", got, err)
			}
			return
		}
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

func TestCheckpointPublicationIsIndependentOfLateStageWrites(t *testing.T) {
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, checkpointCrashRepo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := checkpointCrashCommand(ctx, "", "before publish", dest)
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
	reader := bufio.NewReader(out)
	if line, err := reader.ReadString('\n'); line != "ready\n" || err != nil {
		t.Fatalf("writer did not reach publication: %q, %v", line, err)
	}
	stage := testCheckpointStagePath(dest)
	path := filepath.Join(stage, checkpointCrashShard)
	staged, edited := rewriteCheckpointWeightInPlace(t, path, files[checkpointCrashShard])
	prepared, err := filepath.Glob(filepath.Join(filepath.Dir(stage), "publish-*", checkpointCrashShard))
	if err != nil || len(prepared) != 1 {
		t.Fatalf("publication candidate = %v, %v", prepared, err)
	}
	if candidate, err := os.Stat(prepared[0]); err != nil || os.SameFile(staged, candidate) {
		t.Errorf("candidate shares the staging inode: %v", err)
	}
	payload := SetBytes(readTestCheckpointMarker(t, filepath.Dir(prepared[0])).Files)
	t.Logf("checkpoint payload %d bytes per directory, %d bytes across stage and candidate", payload, 2*payload)
	if err := in.Close(); err != nil {
		t.Fatal(err)
	}
	childOutput, _ := io.ReadAll(reader)
	if err := cmd.Wait(); err != nil {
		t.Fatalf("writer failed: %v\n%s", err, childOutput)
	}
	_, cached := CachedCheckpoint(dest)
	if raw, err := os.ReadFile(filepath.Join(dest, checkpointCrashShard)); err != nil || !bytes.Equal(raw, files[checkpointCrashShard]) || !cached {
		t.Errorf("published bytes = %d bytes, CachedCheckpoint=%t, err=%v", len(raw), cached, err)
	}
	if raw, err := os.ReadFile(path); err != nil || !bytes.Equal(raw, edited) {
		t.Errorf("cleanup changed the edited stage: %d bytes, %v", len(raw), err)
	}
}

func TestCheckpointExplicitRepullRehashesExistingWeight(t *testing.T) {
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, checkpointCrashRepo)
	dest, err := CacheDir(checkpointCrashRepo)
	if err != nil {
		t.Fatal(err)
	}
	pullCrashCheckpoint(t, dest)
	weight := filepath.Join(dest, checkpointCrashShard)
	info, edited := rewriteCheckpointWeightInPlace(t, weight, files[checkpointCrashShard])
	sidecar := sidecarPath(weight)
	if _, err := os.Stat(sidecar); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	if out, err := checkpointCrashCommand(ctx, "", "explicit repull", dest).CombinedOutput(); err != nil {
		t.Fatalf("explicit repull: %v\n%s", err, out)
	}
	if after, err := os.Stat(weight); err != nil || !sameCheckpointFileState(info, after) {
		t.Errorf("repull changed the existing inode or metadata: %v", err)
	}
	if raw, err := os.ReadFile(weight); err != nil || !bytes.Equal(raw, edited) {
		t.Errorf("repull changed the existing contents: %v", err)
	}
	if out, err := checkpointCrashCommand(ctx, "", "damaged cache", dest).CombinedOutput(); err != nil {
		t.Errorf("offline consumers accepted the modified checkpoint: %v\n%s", err, out)
	}
	if _, err := os.Lstat(sidecar); !os.IsNotExist(err) {
		t.Errorf("contradicted sidecar is still present: %v", err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.hits[checkpointCrashShard] != 1 {
		t.Errorf("repull downloaded the existing shard again: %v", h.hits)
	}
}

func TestCheckpointReadOnlyCacheRejectsChangedWeightAcrossProcesses(t *testing.T) {
	if runtime.GOOS != "linux" || os.Geteuid() == 0 {
		t.Skip("requires Linux directory permissions and a non-root test process")
	}
	withCacheRoot(t)
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, checkpointCrashRepo)
	dest, err := CacheDir(checkpointCrashRepo)
	if err != nil {
		t.Fatal(err)
	}
	pullCrashCheckpoint(t, dest)
	weight := filepath.Join(dest, checkpointCrashShard)
	sidecar := sidecarPath(weight)
	record, err := os.ReadFile(sidecar)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dest, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dest, 0o755) })
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	if out, err := checkpointCrashCommand(ctx, "", "valid readonly cache", dest).CombinedOutput(); err != nil {
		t.Fatalf("intact read-only cache as uid %d: %v\n%s", os.Geteuid(), err, out)
	}
	_, edited := rewriteCheckpointWeightInPlace(t, weight, files[checkpointCrashShard])
	for _, mode := range []string{"readonly repull", "damaged cache"} {
		if out, err := checkpointCrashCommand(ctx, "", mode, dest).CombinedOutput(); err != nil {
			t.Errorf("%s as uid %d: %v\n%s", mode, os.Geteuid(), err, out)
		}
	}
	if raw, err := os.ReadFile(weight); err != nil || !bytes.Equal(raw, edited) {
		t.Errorf("read-only verification changed the weight: %v", err)
	}
	if raw, err := os.ReadFile(sidecar); err != nil || !bytes.Equal(raw, record) {
		t.Errorf("read-only verification changed the sidecar: %v", err)
	}
	t.Logf("uid %d verified a 0555 checkpoint without modifying its weight or sidecar", os.Geteuid())
}

func TestCheckpointPublicationUsesCurrentTemplateReceipt(t *testing.T) {
	files := shardedTinyRepo(t)
	files[checkpointCrashTemplate] = []byte("{{ messages }}\n")
	h := &fakeHF{files: files}
	h.serve(t, checkpointCrashRepo)
	dest := filepath.Join(t.TempDir(), "checkpoint")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := checkpointCrashCommand(ctx, "", "after template", dest)
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
	reader := bufio.NewReader(out)
	if line, err := reader.ReadString('\n'); line != "ready\n" || err != nil {
		t.Fatalf("writer did not finish the template: %q, %v", line, err)
	}
	_, edited := rewriteCheckpointWeightInPlace(t, filepath.Join(testCheckpointStagePath(dest), checkpointCrashTemplate), files[checkpointCrashTemplate])
	if err := in.Close(); err != nil {
		t.Fatal(err)
	}
	childOutput, _ := io.ReadAll(reader)
	if err := cmd.Wait(); err != nil {
		t.Fatalf("writer did not reject the modified template: %v\n%s", err, childOutput)
	}
	if _, err := os.Lstat(dest); !os.IsNotExist(err) {
		t.Errorf("writer left a published checkpoint: %v", err)
	}
	path := filepath.Join(testCheckpointStagePath(dest), checkpointCrashTemplate)
	if raw, err := os.ReadFile(path); err != nil || !bytes.Equal(raw, edited) {
		t.Errorf("failed publication changed staging: %q, %v", raw, err)
	}
	pullCrashCheckpoint(t, dest)
	if raw, err := os.ReadFile(filepath.Join(dest, checkpointCrashTemplate)); err != nil || !bytes.Equal(raw, files[checkpointCrashTemplate]) {
		t.Errorf("repull published wrong template: %q, %v", raw, err)
	}
}

func TestCheckpointReclaimsOnlyVerifiedAbandonedPublication(t *testing.T) {
	for _, mode := range []string{"unchanged", "extra file", "edited payload", "edited sidecar"} {
		t.Run(mode, func(t *testing.T) {
			files := shardedTinyRepo(t)
			h := &fakeHF{files: files}
			h.serve(t, checkpointCrashRepo)
			dest := filepath.Join(t.TempDir(), "checkpoint")
			work := filepath.Dir(testCheckpointStagePath(dest))
			killCheckpointAtTransferBoundary(t, "before publish", dest, nil)
			first, err := filepath.Glob(filepath.Join(work, "publish-*"))
			if err != nil || len(first) != 1 {
				t.Fatalf("expected one abandoned candidate: %v, %v", first, err)
			}
			unowned := filepath.Join(work, "publish-unrecorded")
			if err := os.Mkdir(unowned, 0o700); err != nil {
				t.Fatal(err)
			}
			preserved := map[string][]byte{filepath.Join(unowned, "notes.txt"): []byte("keep")}
			if mode == "extra file" {
				preserved[filepath.Join(first[0], "local.txt")] = []byte("keep")
			}
			for path, data := range preserved {
				if err := os.WriteFile(path, data, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "edited payload" {
				path := filepath.Join(first[0], checkpointCrashShard)
				_, preserved[path] = rewriteCheckpointWeightInPlace(t, path, files[checkpointCrashShard])
			}
			if mode == "edited sidecar" {
				path := sidecarPath(filepath.Join(first[0], checkpointCrashShard))
				raw, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				changed := append(bytes.TrimSuffix(raw, []byte("}")), []byte(`,"owner_note":"keep this record"}`)...)
				if err := os.WriteFile(path, changed, 0o600); err != nil {
					t.Fatal(err)
				}
				preserved[path] = changed
			}
			killCheckpointAtTransferBoundary(t, "before publish", dest, nil)
			if mode == "unchanged" {
				if _, err := os.Lstat(first[0]); !os.IsNotExist(err) {
					t.Errorf("first verified copy remains after second interruption: %v", err)
				}
			}
			pullCrashCheckpoint(t, dest)
			for path, data := range preserved {
				if raw, err := os.ReadFile(path); err != nil || !bytes.Equal(raw, data) {
					t.Errorf("unowned contents changed: %d bytes, %v", len(raw), err)
				}
			}
			left, err := filepath.Glob(filepath.Join(work, "publish-*"))
			if err != nil || len(left) != len(preserved) {
				t.Errorf("unexpected candidates left after recovery: %v, %v", left, err)
			}
		})
	}
}

func rewriteCheckpointWeightInPlace(t *testing.T, path string, contents []byte) (os.FileInfo, []byte) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	edited := bytes.Clone(contents)
	edited[len(edited)-1] ^= 1
	if err := os.WriteFile(path, edited, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if after, err := os.Stat(path); err != nil || !sameCheckpointFileState(info, after) {
		t.Fatalf("failed to preserve inode, size and mtime: %v", err)
	}
	return info, edited
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

func TestCheckpointDefersCleanupForLegacyWriterWithoutLinks(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("requires a filesystem change time")
	}
	for _, mode := range []string{"completed", "partial", "replacement"} {
		t.Run(mode, func(t *testing.T) {
			const extra = "tokenizer.json"
			const interrupted = "tokenizer_config.json"
			const generation = "generation_config.json"
			a, b := shardedTinyRepo(t), shardedTinyRepo(t)
			b[interrupted] = []byte(`{"revision":"B"}`)
			added := interrupted + ".part"
			want := b[interrupted][:len(b[interrupted])/2]
			if mode == "completed" {
				b[extra] = []byte(`{"vocab":{"new":1}}`)
				added, want = extra, b[extra]
			}
			if mode == "replacement" {
				a[generation], b[generation] = []byte(`{"eos_token_id":17}`), []byte(`{"eos_token_id":23}`)
				added, want = generation, b[generation]
			}
			h := &fakeHF{files: a}
			h.serve(t, checkpointCrashRepo)
			parent := t.TempDir()
			dest, backup := filepath.Join(parent, "checkpoint"), filepath.Join(parent, "backup")
			killCheckpointAtTransferBoundary(t, "published without links", dest, nil)
			if err := os.Rename(dest, backup); err != nil {
				t.Fatal(err)
			}
			stage := testCheckpointStagePath(dest)
			newFile := filepath.Join(stage, added)
			var previous os.FileInfo
			var receipt string
			if mode == "replacement" {
				var err error
				previous, err = os.Stat(newFile)
				if err != nil {
					t.Fatal(err)
				}
				receipt = testSealedCheckpointReceipt(t, stage, generation)
			}
			if legacy := os.Getenv("GOINFER_TEST_LEGACY_NOLINK_PULL_BINARY"); legacy != "" {
				setCrashRepoFiles(h, b, interrupted)
				if mode != "partial" {
					h.mu.Lock()
					h.failOnce, h.rejectOnce = "", interrupted
					h.mu.Unlock()
				}
				killCheckpointAtTransferBoundary(t, "older reuse without links", dest, nil)
			} else if mode == "replacement" {
				root, err := os.OpenRoot(stage)
				if err != nil {
					t.Fatal(err)
				}
				defer root.Close()
				if err := replaceRootFile(root, generation, want); err != nil {
					t.Fatal(err)
				}
				if err := os.RemoveAll(receipt); err != nil {
					t.Fatal(err)
				}
			} else {
				// Older writers leave both completed files and partials unrecorded
				// when linking is unavailable, without changing the publication token.
				if err := os.WriteFile(newFile, want, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if raw, err := os.ReadFile(newFile); err != nil || !bytes.Equal(raw, want) {
				t.Fatalf("legacy B did not stage %s: %v", added, err)
			}
			if mode == "replacement" {
				current, err := os.Stat(newFile)
				if err != nil || os.SameFile(previous, current) || current.Size() != previous.Size() {
					t.Fatalf("B did not replace the generation config with a same-size file: %v", err)
				}
				if _, err := os.Lstat(receipt); !os.IsNotExist(err) {
					t.Fatalf("B did not remove the old generation receipt: %v", err)
				}
			}
			if err := os.Rename(backup, dest); err != nil {
				t.Fatal(err)
			}
			setCrashRepoFiles(h, a, "")
			pullCrashCheckpoint(t, dest)
			for _, shard := range []string{checkpointCrashShard, checkpointCrashTail} {
				if raw, err := os.ReadFile(filepath.Join(stage, shard)); err != nil || !bytes.Equal(raw, b[shard]) {
					t.Errorf("retry of A removed B's reusable shard %s: %v", shard, err)
				}
			}
			if raw, err := os.ReadFile(newFile); err != nil || !bytes.Equal(raw, want) {
				t.Errorf("retry of A changed the file staged by B: %v", err)
			}
			if mode != "partial" {
				if err := os.Rename(dest, backup); err != nil {
					t.Fatal(err)
				}
				setCrashRepoFiles(h, b, "")
				pullCrashCheckpoint(t, dest)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.hits[checkpointCrashShard] != 1 || h.hits[checkpointCrashTail] != 1 {
				t.Errorf("B refetched shared shards: %v", h.hits)
			}
		})
	}
}

func testSealedCheckpointReceipt(t *testing.T, stage, name string) string {
	t.Helper()
	root, err := os.OpenRoot(stage)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	record, err := readCheckpointCleanup(root)
	if err != nil {
		t.Fatal(err)
	}
	for dir, state := range record.Transfers {
		if state.Path == name {
			path := filepath.Join(filepath.Dir(stage), dir)
			if _, err := os.Stat(filepath.Join(path, "transfer.json")); err != nil {
				t.Fatal(err)
			}
			return path
		}
	}
	t.Fatalf("no sealed receipt for %s", name)
	return ""
}

func TestCheckpointPullWhenPartialLinkUnavailable(t *testing.T) {
	link := linkCheckpointTransfer
	linkCheckpointTransfer = func(*os.Root, string, string) error { return os.ErrPermission }
	t.Cleanup(func() { linkCheckpointTransfer = link })
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
	if mode == "older reuse without links" {
		legacy = os.Getenv("GOINFER_TEST_LEGACY_NOLINK_PULL_BINARY")
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
