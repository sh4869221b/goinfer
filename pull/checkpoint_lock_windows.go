package pull

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

func tryCheckpointLock(f *os.File) (bool, error) {
	var position windows.Overlapped
	err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &position)
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return false, nil
	}
	return err == nil, err
}
