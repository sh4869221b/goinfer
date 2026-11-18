//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package pull

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

func tryCheckpointLock(f *os.File) (bool, error) {
	err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return false, nil
	}
	return err == nil, err
}
