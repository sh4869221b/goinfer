//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows

package pull

import (
	"fmt"
	"os"
	"runtime"
)

func tryCheckpointLock(*os.File) (bool, error) {
	return false, fmt.Errorf("checkpoint download locking is unavailable on %s", runtime.GOOS)
}
