//go:build !unix && !windows

package pull

import "os"

func checkpointFileID(*os.File, os.FileInfo) string { return "" }
