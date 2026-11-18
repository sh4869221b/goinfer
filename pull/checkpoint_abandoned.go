package pull

import (
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

type checkpointPublicationCandidate struct {
	Name         string            `json:"name"`
	FileID       string            `json:"file_id"`
	MarkerSHA256 string            `json:"marker_sha256"`
	Sidecars     map[string]string `json:"sidecars,omitempty"`
}

func checkpointPublicationDirectoryID(root *os.Root) string {
	f, err := root.Open(".")
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ""
	}
	return checkpointFileID(f, info)
}

// The destination lock is held. An abandoned copy is eligible only if it was
// sealed, still has the recorded identity and contains only its verified data.
func (w *checkpointWorkspace) reclaimAbandonedPublication(stage *os.Root) {
	cleanup, err := readCheckpointCleanup(stage)
	if err != nil || cleanup.Owner != w.owner || !w.matchesPublication(stage, cleanup.Token) {
		return
	}
	candidate := cleanup.Candidate
	if candidate == nil || candidate.FileID == "" || !strings.HasPrefix(candidate.Name, "publish-") || !safeRepoPath(candidate.Name) || filepath.Base(candidate.Name) != candidate.Name {
		return
	}
	publish, err := openCheckpointStaging(w.root, candidate.Name)
	if err != nil {
		return
	}
	defer publish.Close()
	if checkpointPublicationDirectoryID(publish) != candidate.FileID {
		return
	}
	entries, ok := candidate.verifiedContents(publish, cleanup.Token)
	if !ok {
		return
	}
	if info, err := w.root.Lstat(candidate.Name); err != nil || !sameCheckpointDirectory(publish, info) {
		return
	}
	for _, entry := range slices.Backward(entries) {
		info, err := publish.Lstat(entry.path)
		if err != nil || !os.SameFile(entry.info, info) {
			return
		}
		if !info.IsDir() && (!sameCheckpointFileState(entry.info, info) || checkpointChangeTime(entry.info) != checkpointChangeTime(info)) {
			return
		}
		if err := publish.Remove(entry.path); err != nil {
			return
		}
	}
	if info, err := w.root.Lstat(candidate.Name); err == nil && sameCheckpointDirectory(publish, info) {
		_ = w.root.Remove(candidate.Name)
	}
}

type checkpointCandidateEntry struct {
	path string
	info os.FileInfo
}

func (c checkpointPublicationCandidate) verifiedContents(root *os.Root, token string) ([]checkpointCandidateEntry, bool) {
	f, err := openRegularDownloadFile(root, checkpointMarker, os.O_RDONLY)
	if err != nil {
		return nil, false
	}
	markerInfo, statErr := f.Stat()
	raw, err := io.ReadAll(io.LimitReader(f, maxIndexBytes+1))
	f.Close()
	if statErr != nil || err != nil || len(raw) > maxIndexBytes || checkpoint.ConfigDigest(raw) != c.MarkerSHA256 {
		return nil, false
	}
	var marker markerDoc
	if err := json.Unmarshal(raw, &marker); err != nil || marker.Publication != token || len(marker.Files) == 0 {
		return nil, false
	}
	files, sidecars := make(map[string]File), make(map[string]File)
	dirs := make(map[string]bool)
	for _, file := range marker.Files {
		path := filepath.FromSlash(file.Path)
		if !safeRepoPath(file.Path) || !filepath.IsLocal(path) || path == checkpointMarker || file.SHA256 == "" {
			return nil, false
		}
		files[path], sidecars[sidecarPath(path)] = file, file
		for dir := filepath.Dir(path); dir != "."; dir = filepath.Dir(dir) {
			dirs[dir] = true
		}
	}
	entries := []checkpointCandidateEntry{{path: checkpointMarker, info: markerInfo}}
	var digestEntries []checkpointCandidateEntry
	err = fs.WalkDir(root.FS(), ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || path == "." {
			return err
		}
		path = filepath.FromSlash(path)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if path == checkpointMarker && sameCheckpointFileState(markerInfo, info) {
			return nil // keep the ownership evidence until all other entries are removed
		}
		file, payload := files[path]
		source, sidecar := sidecars[path]
		switch {
		case info.IsDir() && dirs[path]:
		case info.Mode().IsRegular() && payload && verifyAbandonedFile(root, path, file, info):
		case info.Mode().IsRegular() && sidecar && verifyAbandonedSidecar(root, path, source, info, c.Sidecars[filepath.ToSlash(path)]):
			digestEntries = append(digestEntries, checkpointCandidateEntry{path: path, info: info})
			return nil
		default:
			return fs.ErrInvalid
		}
		entries = append(entries, checkpointCandidateEntry{path: path, info: info})
		return nil
	})
	// Reverse removal drops sidecars before their payloads so reaping can resume.
	return append(entries, digestEntries...), err == nil
}

func verifyAbandonedFile(root *os.Root, path string, file File, info os.FileInfo) bool {
	f, err := openRegularDownloadFile(root, path, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer f.Close()
	opened, err := f.Stat()
	return err == nil && sameCheckpointFileState(info, opened) && checkOpenCheckpointDigest(root, path, file, f, info) == nil
}

func verifyAbandonedSidecar(root *os.Root, path string, source File, info os.FileInfo, expected string) bool {
	if expected == "" || info.Size() > 4096 {
		return false
	}
	record, err := openRegularDownloadFile(root, path, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer record.Close()
	observed, err := record.Stat()
	if err != nil || !sameCheckpointFileState(info, observed) {
		return false
	}
	raw, err := io.ReadAll(io.LimitReader(record, 4097))
	if err != nil || len(raw) > 4096 || checkpoint.ConfigDigest(raw) != expected {
		return false
	}
	var digest digestSidecar
	if err := json.Unmarshal(raw, &digest); err != nil {
		return false
	}
	name := filepath.FromSlash(source.Path)
	f, err := openRegularDownloadFile(root, name, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || opened.Size() != source.Size {
		return false
	}
	return digest.Size == opened.Size() && digest.ModTime == opened.ModTime().UnixNano() && digest.SHA256 == source.SHA256 && digest.FileID == checkpointFileID(f, opened) && digest.ChangeTime == checkpointChangeTime(opened)
}
