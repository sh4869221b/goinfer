"""Write a compressed real-model oracle using reviewed local PFN source.

Usage: python scripts/pin_plamo3_real.py MODEL_DIR OUTPUT.json.gz
Requires the packages and pinned model revision used by pin_plamo3_tiny.py.
Widening BF16 weights to fp32 isolates runtime math from activation rounding.
"""
import gzip
import json
import pathlib
import sys

import torch
from safetensors.torch import load_file

from pin_plamo3_tiny import module


def main():
    root = pathlib.Path(sys.argv[1])
    pfn = module("pfn_real_reference", root / "modeling_plamo.py")
    tokmod = module("pfn_real_tokenizer", root / "tokenization_plamo.py")
    torch.set_num_threads(4)
    cfg = pfn.Plamo3Config(**json.loads((root / "config.json").read_text()))
    model = pfn.Plamo3ForCausalLM(cfg).eval().float()
    model.load_state_dict(load_file(root / "model.safetensors"))
    tk = tokmod.Plamo3Tokenizer.from_pretrained(str(root), local_files_only=True)
    cases = []
    for text, src, dst in [
        ("The meeting has been moved to next Monday.", "en", "ja"),
        ("この文章を英語に翻訳してください。", "ja", "en"),
    ]:
        prompt = (
            "<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>" + src
            + "<|plamo:msg|>" + text
            + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + dst
            + "<|plamo:msg|>"
        )
        ids = tk.encode(prompt, add_special_tokens=True)
        traces = []
        handles = [
            layer.register_forward_hook(
                lambda _, __, out: traces.append(out[0][0, -1].detach().tolist())
            )
            for layer in model.model.layers.layers
        ]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float().tolist()
        for handle in handles:
            handle.remove()
        cases.append(dict(prompt_ids=ids, logits=logits, hidden=traces))
        print("Reference ready", src, dst, len(ids), flush=True)
    with gzip.open(sys.argv[2], "wt") as f:
        json.dump(cases, f)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
