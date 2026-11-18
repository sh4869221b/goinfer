package pull

import (
	"crypto/rand"
	"fmt"
	"os"
)

func openRegularDownloadFile(root *os.Root, name string, flags int) (*os.File, error) {
	f, err := root.OpenFile(name, flags|downloadFileSafetyFlags, 0o644)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if !info.Mode().IsRegular() {
		f.Close()
		return nil, fmt.Errorf("%s is not a regular file", name)
	}
	return f, nil
}

// replaceRootFile writes through a new regular file so an existing link is replaced as an entry.
func replaceRootFile(root *os.Root, name string, raw []byte) error {
	tmp := ".goinfer-write-" + rand.Text()
	f, err := root.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	defer root.Remove(tmp)
	_, writeErr := f.Write(raw)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	return root.Rename(tmp, name)
}
