//go:build !linux && !darwin

package pull

import "os"

func checkpointChangeTime(os.FileInfo) int64 { return 0 }
