//go:build unix

package pull

import (
	"fmt"
	"os"
	"syscall"
)

func checkpointFileID(_ *os.File, info os.FileInfo) string {
	if stat, ok := info.Sys().(*syscall.Stat_t); ok {
		return fmt.Sprintf("%x:%x", stat.Dev, stat.Ino)
	}
	return ""
}
