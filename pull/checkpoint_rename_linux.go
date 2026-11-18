package pull

import (
	"os"

	"golang.org/x/sys/unix"
)

func renameCheckpointInit(parent *os.Root, src, dest string) error {
	dir, err := parent.Open(".")
	if err != nil {
		return err
	}
	defer dir.Close()
	if err := unix.Renameat2(int(dir.Fd()), src, int(dir.Fd()), dest, unix.RENAME_NOREPLACE); err != nil {
		return &os.LinkError{Op: "initializing checkpoint directory", Old: src, New: dest, Err: err}
	}
	return nil
}
