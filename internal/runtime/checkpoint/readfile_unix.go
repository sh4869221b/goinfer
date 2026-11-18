//go:build unix

package checkpoint

import (
	"os"
	"syscall"
)

// A path can become a FIFO after it is checked. Open without waiting for a writer.
const readFlags = os.O_RDONLY | syscall.O_NONBLOCK
