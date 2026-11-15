"""Load the uneven checkpoint with plain transformers (no trust_remote_code).

The checkpoint is an ordinary Qwen3.5-MoE checkpoint whose routed experts are placeholders, plus one file per layer
(uneven_000.safetensors, ...) with that layer's real router rows and experts of different widths. Loading it with
from_pretrained alone gives the placeholders, not this model: call load() below, or use the -padded repo, which
vLLM and transformers load directly.

    from load_uneven import load
    model, tok = load("<local dir of this repo>")
"""

from __future__ import annotations

import torch


def load(path: str, dtype=torch.bfloat16, device_map="auto"):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    import uneven  # next to this file in the repo

    model = Qwen3_5MoeForCausalLM.from_pretrained(path, dtype=dtype, device_map=device_map)
    kept = uneven.attach(model, path)
    print(f"uneven experts attached: {kept:,} routed neurons ({kept / len(model.model.layers):,.0f} per layer)")
    return model.eval(), AutoTokenizer.from_pretrained(path)
