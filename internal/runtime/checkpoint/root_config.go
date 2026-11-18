package checkpoint

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ReadRootConfig follows config.json within root and reads a bounded regular file.
// A missing config is returned as nil; an unusable link returns an error.
func ReadRootConfig(root *os.Root) ([]byte, error) {
	name := "config.json"
	info, err := root.Lstat(name)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		name, info, err = resolveRootFileLink(root, name)
		if err != nil {
			return nil, err
		}
	}
	return readCheckedRootFile(root, name, info, MaxConfigBytes)
}

// StatRootFile follows a top-level checkpoint file's links within root, including absolute aliases.
func StatRootFile(root *os.Root, name string) (os.FileInfo, error) {
	_, info, err := rootFileInfo(root, name)
	return info, err
}

// ReadRootFile reads up to limit bytes from a top-level regular file inside root.
func ReadRootFile(root *os.Root, name string, limit int64) ([]byte, error) {
	name, info, err := rootFileInfo(root, name)
	if err != nil {
		return nil, err
	}
	return readCheckedRootFile(root, name, info, limit)
}

// OpenRootFile confines relative and absolute link chains to root and opens the checked regular file.
func OpenRootFile(root *os.Root, name string) (*os.File, error) {
	if !filepath.IsLocal(name) {
		return nil, fmt.Errorf("file path %q is outside the checkpoint directory", name)
	}
	name, info, err := resolveRootFileLink(root, name)
	if err != nil {
		return nil, err
	}
	return openCheckedRootFile(root, name, info)
}

func rootFileInfo(root *os.Root, name string) (string, os.FileInfo, error) {
	info, err := root.Lstat(name)
	if err != nil || info.Mode()&os.ModeSymlink == 0 {
		return name, info, err
	}
	return resolveRootFileLink(root, name)
}

func readCheckedRootFile(root *os.Root, name string, info os.FileInfo, limit int64) ([]byte, error) {
	if info.Size() > limit {
		return nil, fmt.Errorf("%s exceeds %d bytes", name, limit)
	}
	f, err := openCheckedRootFile(root, name, info)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, limit+1))
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("%s exceeds %d bytes", name, limit)
	}
	return raw, err
}

func openCheckedRootFile(root *os.Root, name string, info os.FileInfo) (*os.File, error) {
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", name)
	}
	f, err := root.OpenFile(name, readFlags, 0)
	if err != nil {
		return nil, err
	}
	opened, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		f.Close()
		return nil, fmt.Errorf("%s changed while opening", name)
	}
	return f, nil
}

// Expand each component through root so an absolute link can be converted to a rooted name at any hop.
func resolveRootFileLink(root *os.Root, name string) (string, os.FileInfo, error) {
	const maxLinks = 40
	pending := strings.Split(filepath.ToSlash(name), "/")
	var parents []string
	links := 0
	for len(pending) != 0 {
		part := pending[0]
		pending = pending[1:]
		switch part {
		case "", ".":
			continue
		case "..":
			if len(parents) == 0 {
				return "", nil, fmt.Errorf("relative link leaves the checkpoint directory")
			}
			parents = parents[:len(parents)-1]
			continue
		}
		name := filepath.Join(append(parents, part)...)
		info, err := root.Lstat(name)
		if err != nil {
			return "", nil, err
		}
		if info.Mode()&os.ModeSymlink == 0 {
			if len(pending) == 0 {
				return name, info, nil
			}
			if !info.IsDir() {
				return "", nil, fmt.Errorf("link component %q is not a directory", name)
			}
			parents = append(parents, part)
			continue
		}
		links++
		if links > maxLinks {
			return "", nil, fmt.Errorf("link exceeds %d symlink hops", maxLinks)
		}
		target, err := root.Readlink(name)
		if err != nil {
			return "", nil, err
		}
		if filepath.IsAbs(target) {
			target, err = rootedFileTarget(root, target, len(pending) != 0)
			if err != nil {
				return "", nil, err
			}
			parents = nil
		} else if filepath.VolumeName(target) != "" || strings.HasPrefix(filepath.ToSlash(target), "/") {
			return "", nil, fmt.Errorf("link has no local relative target")
		}
		pending = append(strings.Split(filepath.ToSlash(target), "/"), pending...)
	}
	return "", nil, fmt.Errorf("link target is not a regular file")
}

// Match the first absolute directory prefix with the identity of the open root. Once inside,
// the remainder is resolved through root, including links that might try to leave it again.
func rootedFileTarget(root *os.Root, target string, directory bool) (string, error) {
	if filepath.Clean(target) != target {
		return "", fmt.Errorf("absolute link is not canonical")
	}
	rootInfo, err := root.Stat(".")
	if err != nil {
		return "", err
	}
	var prefixes []string
	lastDir := filepath.Dir(target)
	if directory {
		lastDir = target
	}
	for dir := lastDir; ; dir = filepath.Dir(dir) {
		prefixes = append(prefixes, dir)
		if filepath.Dir(dir) == dir {
			break
		}
	}
	for i := len(prefixes) - 1; i >= 0; i-- {
		info, err := os.Stat(prefixes[i])
		if os.IsNotExist(err) {
			break // no prefix reached the open checkpoint root
		}
		if err != nil {
			return "", err
		}
		if os.SameFile(rootInfo, info) {
			return filepath.Rel(prefixes[i], target)
		}
	}
	return "", fmt.Errorf("absolute link leaves the checkpoint directory")
}
