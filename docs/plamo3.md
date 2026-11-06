# Experimental PLaMo 3 Translate support

This branch adds the PLaMo 3 translation decoder and tokenizer. It does not add PLaMo 2 support: PLaMo 2 Translate uses a different Mamba hybrid architecture.

The implementation follows the official [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b/tree/cc8e88169807d5a272e8ff06db12eb0129b67f81) source, pinned at `cc8e88169807d5a272e8ff06db12eb0129b67f81`.

## Run a translation

Download the original BF16 checkpoint into a local directory with Hugging Face's CLI:

```sh
hf download pfnet/plamo-3-translate-0.8b \
  --revision cc8e88169807d5a272e8ff06db12eb0129b67f81 \
  --local-dir ./models/plamo-3-translate-0.8b
go run ./demo/plamo-translate \
  -model ./models/plamo-3-translate-0.8b \
  -source-lang en -target-lang ja \
  -text 'The meeting has been moved to next Monday.'
```

The example uses CPU inference, runtime int4 quantization, greedy decoding and the checkpoint's generation stop tokens. `-quant bf16` keeps the original weight precision; `-quant f32` widens it. Python is only needed to fetch weights with the example download command and reproduce the reference fixtures, not for Go inference. The Go model/tokenizer loaders execute no Python checkpoint code.

The model directory must contain `config.json`, the original safetensors file(s), `generation_config.json`, `tokenizer.jsonl` and `tokenizer_config.json`. `chat_template.jinja` is optional for the example, which explicitly renders the translation prompt. General chat templates and server chat endpoints are not part of this change. Input content is encoded separately from structural markers using `EncodeSegments`.

## Implementation and checks

- QKV and gate/up matrices are split at load. Norm offsets are folded into copied weights: 1 for pre-norms/QK/final norm, 0.2 for the post-attention norm and 1/(5^1.5) for the post-MLP norm.
- Sliding attention and full attention use distinct RoPE bases. Full attention uses the official non-truncated YaRN correction ramp and concentration factor.
- The tokenizer uses right-to-left dynamic programming, integer-rounded unigram scores, longest-prefix tie resolution, raw whitespace and byte fallback. PFN's special-marker, repeated-character and consecutive-space boundary rules are preserved.
- `TestPlamo3Parity` compares every position's logits and both layer outputs against a small random checkpoint run through PFN's Python implementation. It crosses the local window and checks greedy continuation through both single-token and public generation paths.
- `TestPlamoOfficialTokenizerParity` compares 217 multilingual and randomized strings, with and without BOS, against PFN token IDs. Run it with `PLAMO3_TOKENIZER_DIR` pointing at the pinned model directory. The ordinary tokenizer test also covers byte fallback, streaming and separation of input content from control tokens.

Reproduce the fixtures with `scripts/pin_plamo3_tiny.py` using the local official source directory. The script needs torch, transformers 4.57.1, safetensors and numba. It creates random weights, not a subset of the released checkpoint.

```sh
python scripts/pin_plamo3_tiny.py ./models/plamo-3-translate-0.8b
go test ./decoder -run '^TestPlamo3' -v
PLAMO3_TOKENIZER_DIR=/absolute/path/to/plamo-3-translate-0.8b \
  go test ./tokenizer -run '^TestPlamo' -v
```

Use an absolute path for `PLAMO3_TOKENIZER_DIR`, since Go tests run in the package directory.

The real 0.8B checkpoint was also checked on Linux amd64 CPU: two 26-token English/Japanese prompts matched the reference at all 16 layers (cosine at least 0.99999), with matching next-token argmax and final-logit cosine reported as 1.000000000. Go used BF16 weights widened for FP32 arithmetic; the reference explicitly widened the released weights to FP32. This checks the original weights and both input languages, not broad translation quality or long context. Runtime int4 generation produced these smoke-test translations:

| Source | Output |
| --- | --- |
| The meeting has been moved to next Monday. | 会議は来週の月曜日に変更になりました。 |
| 会議は来週の月曜日に変更されました。 | The meeting has been moved to next Monday. |

Reproduce the real-model comparison:

```sh
python scripts/pin_plamo3_real.py /absolute/path/to/plamo-3-translate-0.8b /tmp/plamo3-real.json.gz
PLAMO3_MODEL_DIR=/absolute/path/to/plamo-3-translate-0.8b \
PLAMO3_REAL_GOLDEN=/tmp/plamo3-real.json.gz \
  go test -tags realckpt ./decoder -run '^TestPlamo3RealOracle$' -v
```

The downloaded safetensors SHA-256 was `7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5`, matching Hugging Face's declared digest for the pinned revision. Reference packages were torch 2.15.0+cpu, transformers 4.57.1 and numba 0.68.1.

## Scope and remaining validation

Supported input is the original unquantized safetensors layout. Pre-quantized FP8/W4/NVFP4 checkpoints and PLaMo GGUF files are unsupported. Standalone tokenizer packaging inside `.giw` and the built-in Hugging Face `pull` workflow are not implemented for tokenizer.jsonl; use the local directory and dedicated example.

The shared descriptor can express the other PLaMo 3 sizes, but 2B, 8B and 31B have not been run here. CUDA, Metal, the user's RTX 4070, long-context limits and translation quality benchmarks have not been validated. No performance or VRAM-fit claim is made. The capability matrix's parity status remains pending until full model oracle validation.

GoInfer is MIT licensed. Model weights and PFN source use the [PLaMo Community License](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/cc8e88169807d5a272e8ff06db12eb0129b67f81/LICENSE/en), with separate terms for commercial use; the Go implementation does not redistribute model weights or PFN source. Follow those terms when downloading or deploying the model.
