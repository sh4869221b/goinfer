//go:build !unix

package checkpoint

import "os"

const readFlags = os.O_RDONLY
