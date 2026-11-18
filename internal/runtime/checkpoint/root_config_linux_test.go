package checkpoint

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestCheckedConfigOpenRejectsFIFOReplacement(t *testing.T) {
	const child = "GOINFER_TEST_CONFIG_FIFO_CHILD"
	if os.Getenv(child) != "" {
		dir := t.TempDir()
		const name = "config.json"
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
		root, err := os.OpenRoot(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer root.Close()
		info, err := root.Lstat(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := syscall.Mkfifo(path, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := readCheckedRootFile(root, name, info, MaxConfigBytes); err == nil || !strings.Contains(err.Error(), "changed while opening") {
			t.Fatalf("expected FIFO rejection, got %v", err)
		}
		return
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCheckedConfigOpenRejectsFIFOReplacement$")
	cmd.Env = append(os.Environ(), child+"=1")
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		t.Fatalf("opening a FIFO without a writer blocked: %s", out)
	}
	if err != nil {
		t.Fatalf("FIFO check failed: %v\n%s", err, out)
	}
}
