//go:build !linux && !darwin

package pull

import "os"

func renameCheckpointInit(parent *os.Root, src, dest string) error {
	if _, err := parent.Lstat(dest); err == nil {
		return &os.PathError{Op: "initializing checkpoint directory", Path: dest, Err: os.ErrExist}
	} else if !os.IsNotExist(err) {
		return err
	}
	return parent.Rename(src, dest)
}
