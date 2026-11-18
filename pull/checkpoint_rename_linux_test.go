package pull

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func TestCheckpointInitializationRequiresNoReplace(t *testing.T) {
	const dest = "checkpoint"
	name := checkpointWorkspaceName(dest)
	for _, mode := range []string{"unsupported", "concurrent directory"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			parent, err := os.OpenRoot(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer parent.Close()
			if err := parent.WriteFile("notes.txt", []byte("keep"), 0o600); err != nil {
				t.Fatal(err)
			}
			rename := checkpointRenameNoReplace
			t.Cleanup(func() { checkpointRenameNoReplace = rename })
			checkpointRenameNoReplace = func(oldFD int, old string, newFD int, new string, flags uint) error {
				if flags != unix.RENAME_NOREPLACE || new != name {
					t.Fatalf("unexpected initialization rename to %q with flags %d", new, flags)
				}
				if mode == "unsupported" {
					return unix.EOPNOTSUPP
				}
				// An empty directory would be replaced by a plain rename.
				if err := parent.Mkdir(name, 0o700); err != nil {
					t.Fatal(err)
				}
				return rename(oldFD, old, newFD, new, flags)
			}
			work, err := lockCheckpointDestination(parent, dest)
			if err == nil {
				work.Close()
				t.Fatal("unexpectedly acquired a workspace")
			}
			if mode == "unsupported" {
				if !errors.Is(err, unix.EOPNOTSUPP) || !strings.Contains(err.Error(), "RENAME_NOREPLACE") {
					t.Errorf("expected filesystem guidance, got %v", err)
				}
				if _, err := parent.Lstat(name); !os.IsNotExist(err) {
					t.Errorf("failed initialization left a workspace: %v", err)
				}
			} else {
				if !strings.Contains(err.Error(), "no verified owner") {
					t.Errorf("expected rejection of concurrent workspace, got %v", err)
				}
				if entries, err := os.ReadDir(filepath.Join(dir, name)); err != nil || len(entries) != 0 {
					t.Errorf("concurrently created directory changed: %v, %v", entries, err)
				}
			}
			if raw, err := parent.ReadFile("notes.txt"); err != nil || string(raw) != "keep" {
				t.Errorf("unrelated data changed: %q, %v", raw, err)
			}
			if matches, err := filepath.Glob(filepath.Join(dir, ".goinfer-init-*")); err != nil || len(matches) != 0 {
				t.Errorf("initializer left temporary data: %v, %v", matches, err)
			}
		})
	}
}
