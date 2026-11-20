"""Turn a LoRA from heal_train.py (torch, {"model.layers.N.<path>.A": (r, in), ".B": (out, r)}, scale alpha / r) into an
mlx-lm adapter directory (adapters.safetensors + adapter_config.json) for the 4-bit MLX student, so that
`mlx_lm server --model <mlx model> --adapter-path <dir>` serves the healed model without merging or re-quantizing.

  python to_mlx_adapter.py lora_1e7.pt ../cache/mlx_96x256_q4 ../cache/heal_adapters/1e7 [--scale 2.0]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("lora")
    ap.add_argument("mlx_model")
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=2.0, help="alpha / r used in training")
    args = ap.parse_args()

    sd = torch.load(args.lora, map_location="cpu")
    pairs, experts = {}, {}
    for name, t in sd.items():
        if ".mlp.experts.lora_" in name:  # README 26 B: folded into the build by fold_expert_lora_mlx.py instead
            continue
        m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.(A|B)", name)
        e = re.fullmatch(r"expert_lora\.(\d+)\.(A1|B1|A2|B2)", name)
        assert m or e, f"unexpected parameter {name}"
        if m:
            pairs.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t.float().numpy()
        else:
            experts.setdefault(int(e.group(1)), {})[e.group(2)] = t.float().numpy()
    rank = next(iter(pairs.values()))["A"].shape[0]
    # routed experts (ExpertLoRA: gate_up fused, A1 (E, r, H), B1 (E, 2I, r), A2 (E, r, I), B2 (E, H, r)) become
    # mlx SwitchGLU gate_proj / up_proj / down_proj LoRAs; a smaller rank is zero-padded to the linear rank (exact)
    for i, d in experts.items():
        inter = d["B1"].shape[1] // 2
        pad = lambda a, axis: np.concatenate([a, np.zeros(a.shape[:axis] + (rank - a.shape[axis],) + a.shape[axis + 1:], a.dtype)], axis)
        a1, a2 = pad(d["A1"], 1), pad(d["A2"], 1)
        b1, b2 = pad(d["B1"], 2), pad(d["B2"], 2)
        pairs[(i, "mlp.switch_mlp.gate_proj")] = {"A3": a1, "B3": b1[:, :inter]}
        pairs[(i, "mlp.switch_mlp.up_proj")] = {"A3": a1, "B3": b1[:, inter:]}
        pairs[(i, "mlp.switch_mlp.down_proj")] = {"A3": a2, "B3": b2}
    keys = sorted({path for _, path in pairs})
    model, _ = load(args.mlx_model)
    n_layers = len(model.layers)
    cfg = {"rank": rank, "scale": args.scale, "dropout": 0.0, "keys": keys}
    linear_to_lora_layers(model, n_layers, cfg)
    params = dict(tree_flatten(model.parameters()))
    weights = {}
    for (i, path), ab in pairs.items():
        hits_a = [k for k in params if k.endswith(f"layers.{i}.{path}.lora_a")]
        assert len(hits_a) == 1, (i, path, hits_a)
        ka = hits_a[0]
        kb = ka[: -len("lora_a")] + "lora_b"
        if "A3" in ab:  # per-expert: mlx keeps (E, r, in) and (E, out, r), the same layout as ExpertLoRA
            a, b = ab["A3"], ab["B3"]
        else:
            a, b = ab["A"].T, ab["B"].T  # torch: x @ A.T @ B.T ; mlx: x @ lora_a @ lora_b
        assert tuple(params[ka].shape) == a.shape and tuple(params[kb].shape) == b.shape, (ka, params[ka].shape, a.shape, b.shape)
        weights[ka], weights[kb] = mx.array(a.astype(np.float32)), mx.array(b.astype(np.float32))
    n_lora = sum(1 for k in params if k.endswith("lora_a"))
    assert n_lora == len(pairs), f"MLX made {n_lora} LoRA layers, the checkpoint has {len(pairs)}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": n_layers, "lora_parameters": cfg, "source": str(args.lora)}, indent=1))
    print(f"{len(pairs)} LoRA layers (rank {rank}, keys {keys}) -> {out}")


if __name__ == "__main__":
    main()
