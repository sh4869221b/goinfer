package pull

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/internal/runtime/checkpoint"
)

// A safetensors checkpoint as a pull target (docs/tasks/task-checkpoint-fetch-2026-09.md). GGUF is one
// file; a safetensors checkpoint is a SET: config.json, the tokenizer files, and either
// model.safetensors or an index plus the shards it names. decoder.Load already opens such a directory,
// so this only gets the set onto disk, with the guarantees a single file has: every file
// digest-verified, resumable, and nothing at the final path until the whole set is there. Anonymous
// only: a gated original is declined by CheckAccess before any of this runs.

// CheckpointSelector is the selector that names a repo's safetensors checkpoint ("owner/repo:safetensors"). No GGUF
// quant is called that, so it cannot shadow a quant selector.
const CheckpointSelector = "safetensors"

type checkpointConfig struct {
	ModelType  string `json:"model_type"`
	TorchDtype string `json:"torch_dtype"`
	Dtype      string `json:"dtype"`
}

type checkpointMetadata struct {
	config, generation, index []byte
}

func (m checkpointMetadata) contents(name string) []byte {
	switch name {
	case "config.json":
		return m.config
	case checkpoint.GenerationConfigName:
		return m.generation
	case checkpointIndex:
		return m.index
	default:
		return nil
	}
}

// Plan describes a safetensors checkpoint's metadata, decided before any weight byte moves.
// Its JSON form carries the same metadata. DownloadCheckpoint fetches the config files and any index at use time.
type Plan struct {
	Repo      string
	Files     []File // config, tokenizer and processor files, then the weights; sorted by path
	Bytes     int64  // the sum of Files' sizes: what the transfer costs in disk and bandwidth
	ModelType string // config.json's model_type
	Family    string // the capability-matrix family that loads it
	Dtype     string // config.json's torch_dtype (or dtype), "" if absent
}

// CheckpointDownload downloads a prepared checkpoint using the config files and index read during planning.
type CheckpointDownload func(ctx context.Context, dest string, progress func(done, total int64, file string)) (string, error)

// PrepareCheckpointFor returns the plan and a download function bound to its config files, index and file list.
// A nil loads uses the generative check. The returned metadata can be copied or serialized independently.
func PrepareCheckpointFor(ctx context.Context, repo string, loads func(modelType string) (family string, err error)) (Plan, CheckpointDownload, error) {
	if loads == nil {
		loads = generativeLoads
	}
	plan, metadata, err := planCheckpointFor(ctx, repo, loads)
	if err != nil {
		return Plan{}, nil, err
	}
	snapshot := plan
	snapshot.Files = slices.Clone(plan.Files)
	return plan, func(ctx context.Context, dest string, progress func(done, total int64, file string)) (string, error) {
		return downloadCheckpoint(ctx, snapshot, metadata, dest, progress)
	}, nil
}

// checkpointExtras are the non-weight files a checkpoint plan takes when the repo has them: the tokenizer in its several
// forms, the chat template, generation defaults, the image/video processor configs a vision-language family needs, and
// the sentence-transformers module files an embedding model's pooling is read from (aikit's encoder reads
// 1_Pooling/config.json; without it, pooling falls back to its default).
var checkpointExtras = map[string]bool{
	"config.json": true, checkpoint.GenerationConfigName: true,
	"tokenizer.json": true, "tokenizer_config.json": true, "tokenizer.model": true, "special_tokens_map.json": true,
	"added_tokens.json": true, "vocab.json": true, "merges.txt": true,
	"chat_template.jinja": true, "chat_template.json": true,
	"preprocessor_config.json": true, "processor_config.json": true, "video_preprocessor_config.json": true,
	"modules.json": true, "config_sentence_transformers.json": true, "sentence_bert_config.json": true, "1_Pooling/config.json": true,
}

// generativeLoads is PlanCheckpoint's model_type check: a family in the capability matrix with a safetensors loader.
func generativeLoads(modelType string) (string, error) {
	if fam, ok := FamilyForModelType(modelType); ok {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which this build cannot load from safetensors (docs/capability-matrix.json)", modelType)
}

// EncoderFamily is the family a checkpoint loads as when it is the embedding encoder's (EncoderLoads), not a generative
// model's: it opens with serve --embed-model, not --model.
const EncoderFamily = "nomic_bert encoder"

// EmbeddingGemma2Family is the family an EmbeddingGemma 2 checkpoint loads as (serve --embed-model, through
// goinfer's own embeddinggemma2 package rather than aikit's encoder).
const EmbeddingGemma2Family = "embedding_gemma2 encoder"

// EncoderLoads is the model_type check of goinfer's embedding encoders (serve --embed-model): aikit's encoder.Load,
// which loads a NomicBert (CodeRankEmbed, nomic-embed-text) and checks no model_type itself, and the embeddinggemma2
// package. The one list of what they take: serve's own check and the pull CLI both read it.
func EncoderLoads(modelType string) (string, error) {
	switch modelType {
	case "nomic_bert":
		return EncoderFamily, nil
	case "embedding_gemma2":
		return EmbeddingGemma2Family, nil
	}
	return "", fmt.Errorf("model_type %q, which the embedding encoder does not load (it loads nomic_bert and embedding_gemma2)", modelType)
}

// AnyLoads is the pull CLI's model_type check: a generative family this build loads from safetensors,
// else the embedding encoder. The CLI fetches for whatever will open the files; --model and the web UI,
// which load generative models only, keep PlanCheckpoint's check.
func AnyLoads(modelType string) (string, error) {
	if fam, err := generativeLoads(modelType); err == nil {
		return fam, nil
	}
	if fam, err := EncoderLoads(modelType); err == nil {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which this build loads neither as a generative model from safetensors (docs/capability-matrix.json) nor as an embedding encoder (nomic_bert, embedding_gemma2)", modelType)
}

// safeRepoPath reports whether a repo file path is safe to join under a local directory: relative, no ".." or empty
// segment, no backslash. A tree listing is remote input.
func safeRepoPath(p string) bool {
	if p == "" || strings.HasPrefix(p, "/") || strings.Contains(p, "\\") {
		return false
	}
	for seg := range strings.SplitSeq(p, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
	}
	return true
}

// listTree returns every file in the repo's main branch (recursively), with LFS digests.
func listTree(ctx context.Context, repo string) ([]File, error) {
	resp, err := get(ctx, hfAPI+"/"+repo+"/tree/main?recursive=true")
	if err != nil {
		return nil, fmt.Errorf("listing %s: %w", repo, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("listing %s: HuggingFace returned %s", repo, resp.Status)
	}
	var entries []treeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, fmt.Errorf("parsing file list: %w", err)
	}
	var out []File
	for _, e := range entries {
		if e.Type != "file" {
			continue
		}
		f := File{Path: e.Path, Size: e.Size}
		if e.LFS != nil {
			f.SHA256 = e.LFS.OID
		}
		out = append(out, f)
	}
	return out, nil
}

// fetchCheckpointMetadata reads a bounded config or index and checks the declared size and digest.
func fetchCheckpointMetadata(ctx context.Context, repo string, file File, limit int64) ([]byte, error) {
	resp, err := get(ctx, hfCDN+"/"+repo+"/resolve/main/"+file.Path)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", file.Path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: HuggingFace returned %s", file.Path, resp.Status)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", file.Path, err)
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("%s exceeds %d bytes", file.Path, limit)
	}
	if !checkpoint.ConfigMatches(raw, file.Size, file.SHA256) {
		return nil, fmt.Errorf("repo %s's %s does not match the checkpoint metadata; re-run to refresh the plan", repo, file.Path)
	}
	return raw, nil
}

// matrixFamily is the subset of a capability-matrix row FamilyForModelType reads.
type matrixFamily struct {
	Name       string   `json:"name"`
	ModelTypes []string `json:"model_types"`
	Loaders    string   `json:"loaders"`
	Modality   string   `json:"modality"`
}

// FamilyForModelType finds the capability-matrix family whose model_types include mt and that loads from safetensors
// ("will this load?" answered from data already in the tree). ok is false when this build has no such family.
func FamilyForModelType(mt string) (family string, ok bool) {
	var rows []matrixFamily
	if json.Unmarshal(capabilityMatrixJSON, &rows) != nil {
		return "", false
	}
	for _, r := range rows {
		for _, t := range r.ModelTypes {
			if t == mt && strings.Contains(strings.ToLower(r.Loaders), "safetensors") {
				return r.Name, true
			}
		}
	}
	return "", false
}

// PlanCheckpoint decides what a safetensors checkpoint pull of repo would fetch, before any weight moves. It reads the
// tree and two small files (config.json, and the shard index when there is one), and refuses:
//   - a repo with no config.json or no safetensors weights;
//   - a model_type this build cannot load (declined before the first weight byte);
//   - PLaMo 3, whose tokenizer.jsonl layout this pull workflow does not package;
//   - a shard index that names a file the repo does not have, or an unsafe path.
//
// The caller runs CheckAccess first, as for a GGUF pull.
func PlanCheckpoint(ctx context.Context, repo string) (Plan, error) {
	return PlanCheckpointFor(ctx, repo, generativeLoads)
}

// PlanCheckpointFor is PlanCheckpoint with the caller's model_type check. loads names the family that will load the
// checkpoint, or returns why it cannot, and a refusal still comes after reading config.json and before any weight file.
// serve's -embed-model uses it with its encoder's own check, because an embedding encoder is not a generative family in
// the capability matrix.
func PlanCheckpointFor(ctx context.Context, repo string, loads func(modelType string) (family string, err error)) (Plan, error) {
	plan, _, err := planCheckpointFor(ctx, repo, loads)
	return plan, err
}

func planCheckpointFor(ctx context.Context, repo string, loads func(modelType string) (family string, err error)) (Plan, checkpointMetadata, error) {
	files, err := listTree(ctx, repo)
	if err != nil {
		return Plan{}, checkpointMetadata{}, err
	}
	byPath := map[string]File{}
	for _, f := range files {
		byPath[f.Path] = f
	}
	configFile, ok := byPath["config.json"]
	if !ok {
		return Plan{}, checkpointMetadata{}, fmt.Errorf("repo %s has no config.json: not a safetensors (transformers) checkpoint", repo)
	}
	cfgRaw, cfg, err := fetchCheckpointConfig(ctx, repo, configFile)
	if err != nil {
		return Plan{}, checkpointMetadata{}, err
	}
	fam, err := loads(cfg.ModelType)
	if err != nil {
		return Plan{}, checkpointMetadata{}, fmt.Errorf("repo %s is %w; nothing was downloaded", repo, err)
	}
	if err := checkpoint.CheckLayout(repo, cfg.ModelType); err != nil {
		return Plan{}, checkpointMetadata{}, err
	}
	p := Plan{Repo: repo, ModelType: cfg.ModelType, Family: fam, Dtype: cfg.TorchDtype}
	want := map[string]bool{}
	for name := range checkpointExtras {
		if _, ok := byPath[name]; ok {
			want[name] = true
		}
	}
	metadata := checkpointMetadata{config: cfgRaw}
	weights, err := checkpointWeightPaths(byPath, func() ([]byte, error) {
		raw, err := fetchCheckpointMetadata(ctx, repo, byPath[checkpointIndex], maxIndexBytes)
		metadata.index = raw
		return raw, err
	})
	if err != nil {
		return Plan{}, checkpointMetadata{}, fmt.Errorf("repo %s: %w", repo, err)
	}
	if file, ok := byPath[checkpoint.GenerationConfigName]; ok {
		metadata.generation, err = fetchCheckpointMetadata(ctx, repo, file, checkpoint.MaxConfigBytes)
		if err != nil {
			return Plan{}, checkpointMetadata{}, err
		}
	}
	for _, name := range weights {
		want[name] = true
	}
	for name := range want {
		if !safeRepoPath(name) {
			return Plan{}, checkpointMetadata{}, fmt.Errorf("repo %s lists an unsafe path %q", repo, name)
		}
		f := byPath[name]
		p.Files = append(p.Files, f)
		p.Bytes += f.Size
	}
	sort.Slice(p.Files, func(i, j int) bool { return p.Files[i].Path < p.Files[j].Path })
	return p, metadata, nil
}

func fetchCheckpointConfig(ctx context.Context, repo string, file File) ([]byte, checkpointConfig, error) {
	var cfg checkpointConfig
	raw, err := fetchCheckpointMetadata(ctx, repo, file, checkpoint.MaxConfigBytes)
	if err != nil {
		return nil, cfg, err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, cfg, fmt.Errorf("parsing %s's config.json: %w", repo, err)
	}
	if cfg.ModelType == "" {
		return nil, cfg, fmt.Errorf("repo %s's config.json has no model_type, so goinfer cannot tell what it is", repo)
	}
	if cfg.TorchDtype == "" {
		cfg.TorchDtype = cfg.Dtype
	}
	return raw, cfg, nil
}

// TreeDigest is the digest a registry entry pins for a checkpoint directory: sha256 over one line per planned file, in path
// order, "path<TAB>size<TAB>sha256-or-dash". The weights are LFS files, so their lines carry the digest Hugging Face
// declares; a small non-LFS file (config, tokenizer) carries its size only, which is what the tree listing offers. It is the
// directory's counterpart of a GGUF entry's sha256: the same upstream-re-upload guard, for a set.
func (p Plan) TreeDigest() string {
	files := append([]File(nil), p.Files...)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	h := sha256.New()
	for _, f := range files {
		sum := f.SHA256
		if sum == "" {
			sum = "-"
		}
		fmt.Fprintf(h, "%s\t%d\t%s\n", f.Path, f.Size, sum)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// VerifyPin is the check a registry entry's directory pull makes after planning and before any weight byte moves: the repo's
// files must hash to the tree digest this build pins. An empty pin (the explicit owner/repo:safetensors form) pins nothing.
func (p Plan) VerifyPin(pin string) error {
	if pin == "" {
		return nil
	}
	if got := p.TreeDigest(); got != pin {
		return fmt.Errorf("%s has changed since this build was cut (tree digest %s, this build pins %s); nothing was downloaded. "+
			"Fetch the new files with the explicit form owner/repo:%s, which pins nothing", p.Repo, got, pin, CheckpointSelector)
	}
	return nil
}

// SizeNote is the disk-and-bandwidth line a plan must show before the transfer: the full-precision
// original costs several times a GGUF of the same model, which goinfer would run at the same quant.
func (p Plan) SizeNote() string {
	s := fmt.Sprintf("%d files, %s to download", len(p.Files), humanBytes(p.Bytes))
	switch strings.ToLower(p.Dtype) {
	case "bfloat16", "float16", "float32":
		s += fmt.Sprintf(" (%s weights: the full-precision original; goinfer quantizes at load, so a GGUF q4 of the same model would download about a quarter of this)", p.Dtype)
	}
	return s
}

// checkpointMarker is the file a completed checkpoint directory carries: the plan it was built from, so a later pull or
// Resolve can tell a complete, goinfer-pulled checkpoint from anything else at that path and verify it offline.
const checkpointMarker = checkpoint.MarkerName

type markerDoc struct {
	Repo  string `json:"repo"`
	Files []File `json:"files"`
}

// checkCompletedCheckpoint reports whether dir contains the files of the current plan.
func checkCompletedCheckpoint(dir string, files []File, metadata checkpointMetadata) (bool, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return false, nil
	}
	defer root.Close()
	m, ok := readCheckpointMarker(root)
	if !ok || !slices.EqualFunc(m.Files, files, func(got, want File) bool {
		// Locally computed metadata hashes are independent of the remote tree pin.
		if metadata.contents(want.Path) != nil && want.SHA256 == "" {
			got.SHA256 = ""
		}
		return got == want
	}) {
		return false, nil
	}
	return verifyCachedCheckpoint(dir, root, m, metadata)
}

// CachedCheckpoint is a complete, supported checkpoint in dir, verified offline from its marker.
func CachedCheckpoint(dir string) (string, bool) {
	if ok, _ := checkCachedCheckpoint(dir); ok {
		return dir, true
	}
	return "", false
}

// checkCachedCheckpoint reports an unusable config or unsupported layout separately from a cache miss.
func checkCachedCheckpoint(dir string) (bool, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return false, nil
	}
	defer root.Close()
	m, ok := readCheckpointMarker(root)
	if !ok {
		return false, nil
	}
	return verifyCachedCheckpoint(dir, root, m, checkpointMetadata{})
}

func readCheckpointMarker(root *os.Root) (markerDoc, bool) {
	marker, err := checkpoint.ReadMarker(root)
	if err != nil || len(marker.Files) == 0 {
		return markerDoc{}, false
	}
	m := markerDoc{Repo: marker.Repo, Files: make([]File, 0, len(marker.Files))}
	for _, f := range marker.Files {
		m.Files = append(m.Files, File(f))
	}
	return m, true
}

func verifyCachedCheckpoint(dir string, root *os.Root, m markerDoc, metadata checkpointMetadata) (bool, error) {
	config, err := checkpoint.ReadRootConfig(root)
	if err != nil {
		return false, fmt.Errorf("cached checkpoint %s: config.json must be a readable regular file inside the checkpoint directory: %w", dir, err)
	}
	if metadata.config != nil && !bytes.Equal(config, metadata.config) {
		return false, fmt.Errorf("config.json in %s differs from the current checkpoint plan; use -o <dir> to keep the existing checkpoint", dir)
	}
	if config != nil {
		if err := checkpoint.ValidateConfig(m.Repo, config); err != nil {
			return false, fmt.Errorf("cached checkpoint %s: %w", dir, err)
		}
	}
	if err := verifyCheckpointWeights(root, m.Files, metadata.index); err != nil {
		return false, fmt.Errorf("cached checkpoint %s: %w; move it aside, or pull with -o <dir>", dir, err)
	}
	generation, err := readCachedGeneration(root, m.Files)
	if err != nil {
		return false, fmt.Errorf("cached checkpoint %s: %w", dir, err)
	}
	if metadata.generation != nil && !bytes.Equal(generation, metadata.generation) {
		return false, fmt.Errorf("%s in %s differs from the current checkpoint plan; use -o <dir> to keep the existing checkpoint", checkpoint.GenerationConfigName, dir)
	}
	for _, f := range m.Files {
		switch f.Path {
		case "config.json":
			// Verify the same bytes used for the layout check without following the link again.
			if !checkpoint.ConfigMatches(config, f.Size, f.SHA256) {
				return false, nil
			}
		case checkpointIndex, checkpoint.GenerationConfigName:
			// The metadata readers verify the same bytes used for the layout or generation checks.
		default:
			if _, ok := cachedIntact(filepath.Join(dir, filepath.FromSlash(path.Dir(f.Path))), f); !ok {
				return false, nil
			}
		}
	}
	return true, nil
}

func readCachedGeneration(root *os.Root, files []File) ([]byte, error) {
	for _, file := range files {
		if file.Path == checkpoint.GenerationConfigName {
			return checkpoint.ReadRootGeneration(root, file.Size, file.SHA256, true)
		}
	}
	return checkpoint.ReadRootGeneration(root, 0, "", false)
}

func readCheckpointMetadata(dir, name string) ([]byte, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	return readRootCheckpointMetadata(root, name)
}

func readRootCheckpointMetadata(root *os.Root, name string) ([]byte, error) {
	if name == "config.json" {
		return checkpoint.ReadRootConfig(root)
	}
	if name == checkpoint.GenerationConfigName {
		return checkpoint.ReadRootFile(root, name, checkpoint.MaxConfigBytes)
	}
	return checkpoint.ReadRootFile(root, name, maxIndexBytes)
}

// DownloadCheckpoint checks the current remote config and index against p before fetching weights.
// A Plan supplies metadata, including when it has been decoded from JSON. Use PrepareCheckpointFor
// to download with the exact config and index from the planning request without fetching them again.
func DownloadCheckpoint(ctx context.Context, p Plan, dest string, progress func(done, total int64, file string)) (string, error) {
	p.Files = slices.Clone(p.Files)
	byPath, err := plannedCheckpointFiles(p)
	if err != nil {
		return "", err
	}
	raw, cfg, err := fetchCheckpointConfig(ctx, p.Repo, byPath["config.json"])
	if err != nil {
		return "", err
	}
	if (p.ModelType != "" && p.ModelType != cfg.ModelType) || (p.Dtype != "" && p.Dtype != cfg.TorchDtype) {
		return "", fmt.Errorf("repo %s's config.json model type or dtype differs from the plan; re-plan before downloading", p.Repo)
	}
	if err := checkpoint.CheckLayout(p.Repo, cfg.ModelType); err != nil {
		return "", err
	}
	metadata := checkpointMetadata{config: raw}
	if index, ok := byPath[checkpointIndex]; ok {
		metadata.index, err = fetchCheckpointMetadata(ctx, p.Repo, index, maxIndexBytes)
		if err != nil {
			return "", err
		}
	}
	if generation, ok := byPath[checkpoint.GenerationConfigName]; ok {
		metadata.generation, err = fetchCheckpointMetadata(ctx, p.Repo, generation, checkpoint.MaxConfigBytes)
		if err != nil {
			return "", err
		}
	}
	if _, err := checkpointWeightPaths(byPath, func() ([]byte, error) { return metadata.index, nil }); err != nil {
		return "", fmt.Errorf("repo %s: %w", p.Repo, err)
	}
	return downloadCheckpoint(ctx, p, metadata, dest, progress)
}

func plannedCheckpointFiles(p Plan) (map[string]File, error) {
	byPath := make(map[string]File, len(p.Files))
	for _, f := range p.Files {
		if !safeRepoPath(f.Path) || !filepath.IsLocal(filepath.FromSlash(f.Path)) || f.Path == checkpointMarker {
			return nil, fmt.Errorf("unsafe path %q in the plan for %s", f.Path, p.Repo)
		}
		if _, exists := byPath[f.Path]; exists {
			return nil, fmt.Errorf("checkpoint plan for %s lists %s more than once", p.Repo, f.Path)
		}
		byPath[f.Path] = f
	}
	if _, ok := byPath["config.json"]; !ok {
		return nil, fmt.Errorf("checkpoint plan for %s has no config.json", p.Repo)
	}
	return byPath, nil
}

// downloadCheckpoint stages the config and index alongside resumable downloads, then publishes
// the complete set by renaming the directory. An existing destination must match the plan.
func downloadCheckpoint(ctx context.Context, p Plan, metadata checkpointMetadata, dest string, progress func(done, total int64, file string)) (string, error) {
	byPath, err := plannedCheckpointFiles(p)
	if err != nil {
		return "", err
	}
	for _, name := range []string{"config.json", checkpoint.GenerationConfigName, checkpointIndex} {
		if file, ok := byPath[name]; ok && !checkpoint.ConfigMatches(metadata.contents(name), file.Size, file.SHA256) {
			return "", fmt.Errorf("planned %s does not match the checkpoint files", name)
		}
	}
	complete, err := checkCompletedCheckpoint(dest, p.Files, metadata)
	if err != nil {
		return "", err
	}
	if complete {
		return dest, nil
	}
	if _, err := os.Lstat(dest); err == nil {
		return "", fmt.Errorf("%s exists but is not a complete checkpoint of %s pulled by goinfer: move it aside, or pull with -o <dir>", dest, p.Repo)
	} else if !os.IsNotExist(err) {
		return "", err
	}
	staging := filepath.Clean(dest) + ".partial"
	var need int64
	for _, f := range p.Files {
		if planned := metadata.contents(f.Path); planned != nil {
			if raw, err := readCheckpointMetadata(staging, f.Path); err != nil || !bytes.Equal(raw, planned) {
				need += f.Size // replacements are written beside the stale file before rename
			}
			continue
		}
		need += bytesLeft(f, filepath.Join(staging, filepath.FromSlash(path.Dir(f.Path))))
	}
	if err := checkDiskSpace(p.Repo, need, staging); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(staging), 0o755); err != nil {
		return "", err
	}
	parent, err := os.OpenRoot(filepath.Dir(staging))
	if err != nil {
		return "", err
	}
	defer parent.Close()
	stagingName := filepath.Base(staging)
	if err := parent.Mkdir(stagingName, 0o755); err != nil && !os.IsExist(err) {
		return "", err
	}
	root, err := openCheckpointStaging(parent, stagingName)
	if err != nil {
		return "", fmt.Errorf("opening staging directory %s: %w", staging, err)
	}
	defer root.Close()
	if err := stageCheckpoint(ctx, root, p, metadata, progress); err != nil {
		return "", err
	}
	parentInfo, err := os.Stat(filepath.Dir(staging))
	if err != nil {
		return "", err
	}
	if !sameCheckpointDirectory(parent, parentInfo) {
		return "", fmt.Errorf("checkpoint destination directory changed during download")
	}
	if err := publishCheckpoint(parent, root, stagingName, filepath.Base(dest), p, metadata); err != nil {
		return "", fmt.Errorf("publishing %s: %w", dest, err)
	}
	return dest, nil
}

func stageCheckpoint(ctx context.Context, root *os.Root, p Plan, metadata checkpointMetadata, progress func(done, total int64, file string)) error {
	var done int64
	for _, f := range p.Files {
		var err error
		if planned := metadata.contents(f.Path); planned != nil {
			info, statErr := root.Lstat(f.Path)
			if raw, readErr := readRootCheckpointMetadata(root, f.Path); statErr != nil || !info.Mode().IsRegular() || readErr != nil || !bytes.Equal(raw, planned) {
				err = replaceRootFile(root, f.Path, planned)
			}
		} else {
			name := filepath.FromSlash(f.Path)
			if !reusableStagedFile(root, name, f) {
				if err := root.MkdirAll(filepath.Dir(name), 0o755); err != nil {
					return err
				}
				base := done
				err = downloadFile(ctx, root, name, p.Repo, f, func(d, _ int64) {
					if progress != nil {
						progress(base+d, p.Bytes, f.Path)
					}
				})
			}
		}
		if err != nil {
			return fmt.Errorf("%s (file %s; re-run to resume, the files already verified are kept): %w", p.Repo, f.Path, err)
		}
		done += f.Size
		if progress != nil {
			progress(done, p.Bytes, f.Path)
		}
	}
	return nil
}

// Open the staging directory itself before any transfer writes, then keep cleanup rooted there.
func openCheckpointStaging(parent *os.Root, name string) (*os.Root, error) {
	info, err := parent.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("staging path must be a directory")
	}
	root, err := parent.OpenRoot(name)
	if err != nil {
		return nil, err
	}
	if err := checkCheckpointStaging(root, info); err != nil {
		root.Close()
		return nil, err
	}
	return root, nil
}

func checkCheckpointStaging(root *os.Root, info os.FileInfo) error {
	if !sameCheckpointDirectory(root, info) {
		return fmt.Errorf("staging directory changed")
	}
	return nil
}

func sameCheckpointDirectory(root *os.Root, info os.FileInfo) bool {
	opened, err := root.Stat(".")
	return err == nil && info.IsDir() && os.SameFile(info, opened)
}

func checkpointMarkerFiles(root *os.Root, p Plan, metadata checkpointMetadata) ([]File, error) {
	if err := verifyCheckpointWeights(root, p.Files, metadata.index); err != nil {
		return nil, err
	}
	raw, err := checkpoint.ReadRootConfig(root)
	if err != nil {
		return nil, fmt.Errorf("checking config.json: %w", err)
	}
	if !bytes.Equal(raw, metadata.config) {
		return nil, fmt.Errorf("config.json does not match the checkpoint plan")
	}
	if err := checkpoint.ValidateConfig(p.Repo, raw); err != nil {
		return nil, err
	}
	if metadata.generation == nil {
		if _, err := checkpoint.ReadRootGeneration(root, 0, "", false); err != nil {
			return nil, err
		}
	}
	files := slices.Clone(p.Files)
	for i, f := range files {
		info, err := root.Lstat(filepath.FromSlash(f.Path))
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() || info.Size() != f.Size {
			return nil, fmt.Errorf("staged %s is not a regular file of the recorded size", f.Path)
		}
		if planned := metadata.contents(f.Path); planned != nil {
			if f.Path == checkpoint.GenerationConfigName {
				raw, err := checkpoint.ReadRootGeneration(root, f.Size, f.SHA256, true)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(raw, planned) {
					return nil, fmt.Errorf("staged %s differs from the checkpoint plan", f.Path)
				}
			}
			files[i].SHA256 = checkpoint.ConfigDigest(planned)
		}
	}
	return files, nil
}
