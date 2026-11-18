package pull

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// CacheEntry is one model the pull cache holds (docs/tasks/task-checkpoint-fetch-2026-09.md): what is ON DISK, which is a
// different question from what a server has loaded.
type CacheEntry struct {
	Repo string // owner/repo, from the cache layout (<root>/<owner>/<repo>)
	// Path is what --model, or the web UI's load, takes: the .gguf, a split set's first shard, or the checkpoint
	// directory. For an incomplete entry it is where that file or directory will be once the pull finishes.
	Path     string
	Kind     string // "gguf", "split" or "checkpoint"
	Bytes    int64  // on disk now: the file, every shard present, or the whole directory (its staging copy while incomplete)
	Shards   int    // a split set's shard count; 0 otherwise
	Complete bool   // false for an interrupted pull: a .part file, a split set missing a shard, a checkpoint still in staging
	// Sidecars is the bytes of the .giw sidecars built from this model, beside it: the transcoded copy a load runs from,
	// which costs disk on top of the download.
	Sidecars int64
}

// CacheEntries lists what the pull cache holds, sorted by repo then path. A checkpoint counts as complete when its
// marker covers its weight layout and every file has its recorded size. Listing skips weight digests. A missing
// cache is an empty list.
func CacheEntries() ([]CacheEntry, error) {
	root, err := CacheRoot()
	if err != nil {
		return nil, err
	}
	owners, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []CacheEntry
	for _, o := range owners {
		if !o.IsDir() {
			continue
		}
		repos, err := os.ReadDir(filepath.Join(root, o.Name()))
		if err != nil {
			continue
		}
		for _, r := range repos {
			if !r.IsDir() {
				continue
			}
			dir := filepath.Join(root, o.Name(), r.Name())
			if name, ok := strings.CutSuffix(strings.TrimPrefix(r.Name(), "."), checkpointWorkSuffix); ok && strings.HasPrefix(r.Name(), ".") {
				dest := filepath.Join(root, o.Name(), name)
				if bytes := checkpointStagingBytes(dir, name, dest); bytes > 0 {
					out = append(out, CacheEntry{Repo: o.Name() + "/" + name, Path: dest, Kind: "checkpoint", Bytes: bytes})
				}
				continue
			}
			if name, ok := strings.CutSuffix(r.Name(), ".partial"); ok {
				// Legacy resumable checkpoints used a sibling .partial directory.
				dest := filepath.Join(root, o.Name(), name)
				if _, err := os.Lstat(dest); os.IsNotExist(err) && checkpointStagingBytes(filepath.Join(root, o.Name(), checkpointWorkspaceName(name)), name, dest) == 0 {
					out = append(out, CacheEntry{Repo: o.Name() + "/" + name, Path: dest, Kind: "checkpoint", Bytes: dirBytes(dir)})
				}
				continue
			}
			out = append(out, repoEntries(o.Name()+"/"+r.Name(), dir)...)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Repo != out[j].Repo {
			return out[i].Repo < out[j].Repo
		}
		return out[i].Path < out[j].Path
	})
	return out, nil
}

func checkpointStagingBytes(dir, name, dest string) int64 {
	if _, err := os.Lstat(dest); !os.IsNotExist(err) {
		return 0
	}
	parent, err := os.OpenRoot(filepath.Dir(dir))
	if err != nil {
		return 0
	}
	defer parent.Close()
	root, err := openCheckpointStaging(parent, filepath.Base(dir))
	if err != nil {
		return 0
	}
	defer root.Close()
	owner, err := readCheckpointOwner(root)
	if err != nil || owner.Destination != name || owner.Token == "" {
		return 0
	}
	workspace := checkpointWorkspace{root: root, owner: owner}
	stage, err := workspace.openStage(false)
	if err != nil || stage == nil {
		return 0
	}
	defer stage.Close()
	info, err := stage.Lstat(checkpointOwnerName)
	if err != nil {
		return 0
	}
	metadata := info.Size()
	if publication, err := readCheckpointPublication(stage); err == nil && publication.Owner == owner && publication.Token != "" {
		if info, err := stage.Lstat(checkpointPublicationName); err == nil && info.Mode().IsRegular() {
			metadata += info.Size()
		}
	}
	if cleanup, err := readCheckpointCleanup(stage); err == nil && cleanup.Owner == owner && cleanup.Token != "" {
		if info, err := stage.Lstat(checkpointCleanupName); err == nil && info.Mode().IsRegular() {
			metadata += info.Size()
		}
	}
	return max(dirBytes(stage.Name())-metadata, 0)
}

// repoEntries is one repo directory's models: a checkpoint (the directory itself, when it carries the marker), and the
// GGUF files in it, a split set folded into one entry, with each model's .giw sidecars counted against it.
func repoEntries(repo, dir string) []CacheEntry {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []CacheEntry
	inMarker := map[string]bool{}
	if root, err := os.OpenRoot(dir); err == nil {
		defer root.Close()
		if m, ok := readCheckpointMarker(root); ok {
			complete := verifyCheckpointWeights(root, m.Files, nil) == nil
			if complete {
				_, err := readCachedGeneration(root, m.Files)
				complete = err == nil && verifyCheckpointExtras(root, m.Files) == nil
			}
			for _, f := range m.Files {
				inMarker[filepath.Base(f.Path)] = true
				if complete {
					complete = verifyCheckpointFile(root, f, false) == nil
				}
			}
			out = append(out, CacheEntry{Repo: repo, Path: dir, Kind: "checkpoint", Bytes: dirBytes(dir), Complete: complete})
		}
	}
	sizes := map[string]int64{}
	var giws []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		name := e.Name()
		sizes[name] = info.Size()
		if strings.HasSuffix(name, ".giw") {
			giws = append(giws, name)
		}
	}
	sidecarsFor := func(gguf string) int64 {
		stem := strings.TrimSuffix(gguf, filepath.Ext(gguf)) + "."
		var n int64
		for _, g := range giws {
			if strings.HasPrefix(g, stem) {
				n += sizes[g]
			}
		}
		return n
	}
	sets := map[string][]string{} // a split set's "<prefix>-of-<NNNNN>" -> the shard names present
	for name, size := range sizes {
		if inMarker[name] {
			continue // part of the checkpoint entry above
		}
		lower := strings.ToLower(name)
		switch {
		case strings.HasSuffix(lower, ".gguf.part"):
			final := strings.TrimSuffix(name, ".part")
			if _, done := sizes[final]; !done {
				out = append(out, CacheEntry{Repo: repo, Path: filepath.Join(dir, final), Kind: "gguf", Bytes: size})
			}
		case strings.HasSuffix(lower, ".gguf"):
			if m := shardSuffix.FindStringSubmatch(lower); m != nil {
				key := name[:len(m[1])] + "-of-" + m[3]
				sets[key] = append(sets[key], name)
				continue
			}
			out = append(out, CacheEntry{Repo: repo, Path: filepath.Join(dir, name), Kind: "gguf", Bytes: size, Complete: true, Sidecars: sidecarsFor(name)})
		}
	}
	for _, shards := range sets {
		sort.Strings(shards)
		m := shardSuffix.FindStringSubmatch(strings.ToLower(shards[0]))
		n, _ := strconv.Atoi(m[3]) // five digits, matched by shardSuffix
		var bytes int64
		for _, s := range shards {
			bytes += sizes[s]
		}
		first := shards[0][:len(m[1])] + "-00001-of-" + m[3] + filepath.Ext(shards[0])
		e := CacheEntry{Repo: repo, Path: filepath.Join(dir, first), Kind: "split", Bytes: bytes, Shards: n, Complete: len(shards) == n && shards[0] == first}
		if e.Complete {
			e.Sidecars = sidecarsFor(first)
		}
		out = append(out, e)
	}
	return out
}

// dirBytes is the size of every regular file under dir.
func dirBytes(dir string) int64 {
	var n int64
	_ = filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}
