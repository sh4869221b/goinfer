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
		name, info, err = resolveCachedConfigLink(root, name)
		if err != nil {
			return nil, err
		}
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("config target is not a regular file")
	}
	if info.Size() > MaxConfigBytes {
		return nil, fmt.Errorf("config exceeds %d bytes", MaxConfigBytes)
	}
	return readCheckedConfig(root, name, info)
}

func readCheckedConfig(root *os.Root, name string, info os.FileInfo) ([]byte, error) {
	f, err := root.OpenFile(name, readFlags, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return nil, fmt.Errorf("config target changed while opening")
	}
	raw, err := io.ReadAll(io.LimitReader(f, MaxConfigBytes+1))
	if len(raw) > MaxConfigBytes {
		return nil, fmt.Errorf("config exceeds %d bytes", MaxConfigBytes)
	}
	return raw, err
}

// Expand each component through root so an absolute link can be converted to a rooted name at any hop.
func resolveCachedConfigLink(root *os.Root, name string) (string, os.FileInfo, error) {
	const maxLinks = 40
	pending := []string{name}
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
				return "", nil, fmt.Errorf("relative config link leaves the checkpoint directory")
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
				return "", nil, fmt.Errorf("config link component %q is not a directory", name)
			}
			parents = append(parents, part)
			continue
		}
		links++
		if links > maxLinks {
			return "", nil, fmt.Errorf("config link exceeds %d symlink hops", maxLinks)
		}
		target, err := root.Readlink(name)
		if err != nil {
			return "", nil, err
		}
		if filepath.IsAbs(target) {
			target, err = rootedConfigTarget(root, target, len(pending) != 0)
			if err != nil {
				return "", nil, err
			}
			parents = nil
		} else if filepath.VolumeName(target) != "" || strings.HasPrefix(filepath.ToSlash(target), "/") {
			return "", nil, fmt.Errorf("config link has no local relative target")
		}
		pending = append(strings.Split(filepath.ToSlash(target), "/"), pending...)
	}
	return "", nil, fmt.Errorf("config target is not a regular file")
}

// Match the first absolute directory prefix with the identity of the open root. Once inside,
// the remainder is resolved through root, including links that might try to leave it again.
func rootedConfigTarget(root *os.Root, target string, directory bool) (string, error) {
	if filepath.Clean(target) != target {
		return "", fmt.Errorf("absolute config link is not canonical")
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
		if err != nil {
			return "", err
		}
		if os.SameFile(rootInfo, info) {
			return filepath.Rel(prefixes[i], target)
		}
	}
	return "", fmt.Errorf("absolute config link leaves the checkpoint directory")
}
