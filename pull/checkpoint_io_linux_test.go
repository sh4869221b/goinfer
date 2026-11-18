package pull

import (
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestCheckpointRebuildsMissingWeightSidecarInProcess(t *testing.T) {
	const child = "GOINFER_TEST_CHECKPOINT_IO"
	const repo = "o/tiny"
	const shard = "model-00001-of-00002.safetensors"
	if dir := os.Getenv(child); dir != "" {
		check := func() int64 {
			before := checkpointReadBytes(t)
			if _, ok := CachedCheckpoint(dir); !ok {
				t.Fatal("cache verification failed")
			}
			return checkpointReadBytes(t) - before
		}
		first, second := check(), check()
		info, err := os.Stat(filepath.Join(dir, shard))
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("verified shard size %d bytes; process rchar deltas %d then %d bytes\n", info.Size(), first, second)
		if first < info.Size() || second >= info.Size()/2 {
			t.Fatalf("expected the missing shard digest to be computed once")
		}
		return
	}
	files := shardedTinyRepo(t)
	h := &fakeHF{files: files}
	h.serve(t, repo)
	_, download, err := PrepareCheckpointFor(t.Context(), repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "checkpoint")
	if _, err := download(t.Context(), dest, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(sidecarPath(filepath.Join(dest, shard))); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCheckpointRebuildsMissingWeightSidecarInProcess$")
	cmd.Env = append(os.Environ(), child+"="+dest)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cache reader: %v\n%s", err, out)
	}
	t.Logf("%s", out)
	if _, err := os.Stat(sidecarPath(filepath.Join(dest, shard))); err != nil {
		t.Errorf("verified sidecar was not written: %v", err)
	}
	file := filepath.Join(dest, shard)
	info, err := os.Stat(file)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(dest, "replacement")
	bad := append([]byte(nil), files[shard]...)
	bad[len(bad)-1] ^= 1
	if err := os.WriteFile(replacement, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, file); err != nil {
		t.Fatal(err)
	}
	if _, ok := CachedCheckpoint(dest); ok {
		t.Error("same-size same-mtime replacement reused the previous inode's digest")
	}
}

func checkpointReadBytes(t *testing.T) int64 {
	t.Helper()
	return checkpointIOCount(t, "rchar")
}

func checkpointIOCount(t *testing.T, field string) int64 {
	t.Helper()
	raw, err := os.ReadFile("/proc/self/io")
	if err != nil {
		t.Fatal(err)
	}
	for line := range strings.SplitSeq(string(raw), "\n") {
		if value, ok := strings.CutPrefix(line, field+":"); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			return n
		}
	}
	t.Fatalf("/proc/self/io omits %s", field)
	return 0
}

func TestCheckpointReceiptReadsScaleLinearly(t *testing.T) {
	for _, shards := range []int{16, 32} {
		t.Run(strconv.Itoa(shards), func(t *testing.T) {
			parent, err := os.OpenRoot(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer parent.Close()
			const dest = "checkpoint"
			work, err := lockCheckpointDestination(parent, dest)
			if err != nil {
				t.Fatal(err)
			}
			defer work.Close()
			stage, err := work.openStage(true)
			if err != nil {
				t.Fatal(err)
			}
			defer stage.Close()
			before := checkpointIOCount(t, "syscr")
			contents := []byte("weight")
			digest := fmt.Sprintf("%x", sha256.Sum256(contents))
			for i := range shards {
				name := fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, shards)
				part := name + ".part"
				if err := stage.WriteFile(part, contents, 0o600); err != nil {
					t.Fatal(err)
				}
				info, err := stage.Lstat(part)
				if err != nil {
					t.Fatal(err)
				}
				work.recordPartial(stage, part, info)
				if err := stage.Rename(part, name); err != nil {
					t.Fatal(err)
				}
				work.recordTransfer(stage, File{Path: name, Size: info.Size(), SHA256: digest})
			}
			work.Close()
			recovered, err := lockCheckpointDestination(parent, dest)
			if err != nil {
				t.Fatal(err)
			}
			defer recovered.Close()
			recovered.reclaimTransfers(stage, recovered.snapshotTransfers(stage))
			reads := checkpointIOCount(t, "syscr") - before
			t.Logf("%d shards, %d process read calls", shards, reads)
			if reads > int64(6*shards+12) {
				t.Errorf("expected linear receipt and content reads with fixed overhead, got %d", reads)
			}
			if names, err := fs.ReadDir(stage.FS(), "."); err != nil || len(names) != 1 || names[0].Name() != checkpointOwnerName {
				t.Errorf("expected only the owner record in the stage, got %v, %v", names, err)
			}
		})
	}
}
