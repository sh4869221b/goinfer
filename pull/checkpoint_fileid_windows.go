package pull

import (
	"fmt"
	"os"

	"golang.org/x/sys/windows"
)

func checkpointFileID(f *os.File, _ os.FileInfo) string {
	var stat windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(f.Fd()), &stat); err == nil {
		return fmt.Sprintf("%x:%x%08x", stat.VolumeSerialNumber, stat.FileIndexHigh, stat.FileIndexLow)
	}
	return ""
}
