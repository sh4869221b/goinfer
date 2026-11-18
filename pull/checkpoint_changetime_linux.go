package pull

import (
	"os"
	"syscall"
)

func checkpointChangeTime(info os.FileInfo) int64 {
	if stat, ok := info.Sys().(*syscall.Stat_t); ok {
		return stat.Ctim.Nano()
	}
	return 0
}
