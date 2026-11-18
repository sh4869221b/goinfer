package pull

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

var checkpointRenameNoReplace = unix.Renameat2

func renameCheckpointInit(parent *os.Root, src, dest string) error {
	dir, err := parent.Open(".")
	if err != nil {
		return err
	}
	defer dir.Close()
	if err := checkpointRenameNoReplace(int(dir.Fd()), src, int(dir.Fd()), dest, unix.RENAME_NOREPLACE); err != nil {
		linkErr := &os.LinkError{Op: "initializing checkpoint directory", Old: src, New: dest, Err: err}
		if errors.Is(err, unix.EOPNOTSUPP) || errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EINVAL) {
			// A check followed by plain rename could replace a concurrently created directory.
			return fmt.Errorf("safe checkpoint workspace initialization requires Linux RENAME_NOREPLACE. Choose a destination on a filesystem that supports it: %w", linkErr)
		}
		return linkErr
	}
	return nil
}
