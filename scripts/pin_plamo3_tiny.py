"""Pin a small PLaMo 3 fixture against reviewed, locally downloaded PFN source.

Run with Python containing torch, transformers==4.57.1, safetensors and numba:
  python scripts/pin_plamo3_tiny.py /path/to/plamo-3-translate-0.8b
Source: pfnet/plamo-3-translate-0.8b, revision
cc8e88169807d5a272e8ff06db12eb0129b67f81. No model weights are needed.
"""
import importlib.util
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    source = pathlib.Path(sys.argv[1])
    official = module("pfn_plamo3_reference", source / "modeling_plamo.py")
    torch.set_num_threads(2)
    torch.manual_seed(329)
    cfg = dict(vocab_size=64, hidden_size=64, intermediate_size=128,
               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
               head_dim=128, max_position_embeddings=64, window_size=4,
               sliding_window_pattern=2, rope_theta=1000000, rope_local_theta=10000,
               rope_scaling_factor=256, initial_context_length=4096,
               rms_norm_eps=1e-6, tie_word_embeddings=False, pad_token_id=3,
               bos_token_id=1, eos_token_id=1)
    m = official.Plamo3ForCausalLM(official.Plamo3Config(**cfg)).eval().float()
    # Nonzero norm weights distinguish folding the offset from replacing the weight.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.03)
    prompt = [1, 7, 23, 11, 52, 8, 42, 19, 33, 5, 60, 12]
    traces = []
    handles = [layer.register_forward_hook(
        lambda _, __, result: traces.append(result[0][0].detach().tolist()))
        for layer in m.model.layers.layers]
    with torch.no_grad():
        logits = m(input_ids=torch.tensor([prompt]), use_cache=False).logits[0].tolist()
    for handle in handles:
        handle.remove()
    cur, continuation = list(prompt), []
    with torch.no_grad():
        for _ in range(6):
            nxt = int(m(input_ids=torch.tensor([cur]), use_cache=False).logits[0, -1].argmax())
            continuation.append(nxt)
            cur.append(nxt)
    out = ROOT / "testdata" / "plamo3-tiny"
    out.mkdir(exist_ok=True)
    config = m.config.to_dict()
    config["rope_parameters"] = m.config.rope_parameters
    (out / "config.json").write_text(json.dumps(config, indent=2))
    save_file(m.state_dict(), out / "model.safetensors", metadata={"format": "pt"})
    golden = dict(prompt_ids=prompt, logits=logits, hidden=traces, continuation_ids=continuation)
    (ROOT / "testdata" / "plamo3_tiny_golden.json").write_text(json.dumps(golden))

    tok = module("pfn_plamo3_tokenizer", source / "tokenization_plamo.py")
    ref = tok.Plamo3Tokenizer.from_pretrained(str(source), local_files_only=True)
    strings = ["", "Hello, world!", "会議は来週の月曜日に変更されました。", "日本語をEnglishへ翻訳します。",
               "  a   b    ", "aaaaabbbbb", "a\n\n\nb", "▁literal marker", "🙂𠮷🦄", "é e\u0301 ＡＢＣ",
               "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:msg|>Hello.<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>",
               "<|plamo:unknown_marker|> x", "foo<|plamo:bos|>bar", "x\uee00y", "\x00\t\r\n", "a"*100, " "*100]
    import random
    rng = random.Random(329)
    for _ in range(200):
        strings.append("".join(rng.choice("abcあいう漢字翻訳123  \n▁🙂é!<|>:_") for _ in range(rng.randrange(1, 100))))
    cases = [dict(text=s, ids=ref.encode(s, add_special_tokens=False), bos_ids=ref.encode(s, add_special_tokens=True)) for s in strings]
    (ROOT / "testdata" / "plamo3_tokenizer_golden.json").write_text(json.dumps(cases, ensure_ascii=False))
    print("Pinned layers, logits, greedy continuation and", len(cases), "tokenizer cases")


if __name__ == "__main__":
    main()
