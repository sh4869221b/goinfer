//go:build unix

package pull

import "syscall"

const downloadFileSafetyFlags = syscall.O_NONBLOCK | syscall.O_NOFOLLOW
