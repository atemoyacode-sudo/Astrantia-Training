"""Build 4-bit MLX models of uneven shapes (README 6h) for the Mac, straight from the original model over HTTP.

Everything but the routed experts and their router rows is the same in every cut of the same model, so it is copied
from an existing 4-bit MLX cut (--base). Per layer, the experts that keep a neuron are stored at full width with the
dropped neurons' gate and up rows zeroed. Zero rows stay exactly zero through mx.quantize, so silu(gate) * up is 0 for
them and the model computes what the unpadded one (uneven.py) does; the down projection is left as it is. Parameters
are counted as kept neurons; the file is bigger than that and runs at the original model's speed.

Layers keep different numbers of experts, which stock mlx-lm cannot load: the model type "qwen3_5_moe_uneven"
(mlx_qwen3_5_moe_uneven.py) is installed into this environment's mlx_lm/models/.

  python build_uneven_mlx.py --base ../cache/mlx_96x256_q4 --shapes g128b28160 g128 --out-dir ../cache
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import Source, build_config
from width_pass import parse_uneven, uneven_keep

HERE = Path(__file__).resolve().parent
EXPERT_PARTS = (".mlp.gate.", ".mlp.switch_mlp.")


def install_model_type() -> Path:
    import mlx_lm

    dst = Path(mlx_lm.__file__).parent / "models" / "qwen3_5_moe_uneven.py"
    src = HERE / "mlx_qwen3_5_moe_uneven.py"
    if not dst.exists() or dst.read_text() != src.read_text():
        shutil.copy(src, dst)
    return dst


def main() -> None:
    import mlx.core as mx

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="a 4-bit MLX cut of the same model (for everything but the experts)")
    ap.add_argument("--source", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--stats", default=str(HERE.parent / "cache" / "shape" / "stats.npz"))
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True, help="each shape goes to <out-dir>/mlx_<shape>_q4")
    ap.add_argument("--budget-experts", type=int, default=48)
    ap.add_argument("--layers", type=int, default=0, help="stop after N layers (for a quick test; the model will not load)")
    args = ap.parse_args()
    print("model type installed at", install_model_type(), flush=True)

    base = Path(args.base)
    conf = json.loads((base / "config.json").read_text())
    q = conf["quantization"]
    group, bits = q["group_size"], q["bits"]
    src = Source(args.source)
    cfg = build_config(src.text_config)
    stats = np.load(args.stats)
    specs = {s: parse_uneven(s) for s in args.shapes}
    assert all(specs.values()), "uneven shapes only (g..., t...)"
    outs = {s: Path(args.out_dir) / f"mlx_{s}_q4" for s in specs}
    counts = {s: [] for s in specs}
    kept = {s: [] for s in specs}

    # a base built by this script keeps the non-expert weights in model-rest.safetensors (11-18)
    rest_file = base / "model.safetensors" if (base / "model.safetensors").exists() else base / "model-rest.safetensors"
    rest = {k: v for k, v in mx.load(str(rest_file)).items() if not any(p in k for p in EXPERT_PARTS)}
    for s, out in outs.items():
        out.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out / "model-rest.safetensors"), rest, metadata={"format": "mlx"})
        for f in base.iterdir():
            if f.suffix != ".safetensors" and f.name not in ("config.json", "model.safetensors.index.json", "kl_chat.json", "README.md"):
                shutil.copy(f, out / f.name)
    del rest

    def quant(w: torch.Tensor, n_bits: int) -> dict:
        wq, scales, biases = mx.quantize(mx.array(w.float().numpy()).astype(mx.bfloat16), group, n_bits)
        return {"weight": wq, "scales": scales, "biases": biases}

    keys = ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    fetch = lambda i: {k: src.get(f"layers.{i}.{k}") for k in keys}  # noqa: E731
    pool = ThreadPoolExecutor(1)
    n_layers = args.layers or cfg.num_hidden_layers
    nxt = pool.submit(fetch, 0)
    t0 = time.time()
    for i in range(n_layers):
        lt = time.time()
        sd = nxt.result()
        if i + 1 < n_layers:
            nxt = pool.submit(fetch, i + 1)
        inter = sd["mlp.experts.down_proj"].shape[2]
        for s, spec in specs.items():
            keep = torch.as_tensor(uneven_keep(spec, stats["mass"][i], stats["imp"][i], args.budget_experts * inter))
            et = torch.where(keep.any(1))[0]
            k = keep.index_select(0, et)[:, :, None]
            gu = sd["mlp.experts.gate_up_proj"].index_select(0, et)
            p = f"language_model.model.layers.{i}.mlp"
            parts = {f"{p}.gate": quant(sd["mlp.gate.weight"].index_select(0, et), q.get(f"{p}.gate", {}).get("bits", bits)),
                     f"{p}.switch_mlp.gate_proj": quant(gu[:, :inter] * k, bits),
                     f"{p}.switch_mlp.up_proj": quant(gu[:, inter:] * k, bits),
                     f"{p}.switch_mlp.down_proj": quant(sd["mlp.experts.down_proj"].index_select(0, et), bits)}
            flat = {f"{name}.{part}": v for name, d in parts.items() for part, v in d.items()}
            mx.eval(flat)
            mx.save_safetensors(str(outs[s] / f"model-layer-{i:03d}.safetensors"), flat, metadata={"format": "mlx"})
            counts[s].append(len(et))
            kept[s].append(int(keep.sum()))
            del flat, parts, gu
        del sd
        mx.clear_cache()
        print(f"layer {i + 1}/{n_layers} {time.time() - lt:.0f}s (total {time.time() - t0:.0f}s) experts "
              + " ".join(f"{s}:{counts[s][-1]}" for s in specs), flush=True)

    for s, out in outs.items():
        c = json.loads(json.dumps(conf))
        c["model_type"] = "qwen3_5_moe_uneven"
        tc = c["text_config"]
        routed = sum(kept[s]) * 3 * tc["hidden_size"]
        tc.update({"num_experts": tc["num_experts_per_tok"], "moe_intermediate_size": inter, "uneven_experts": counts[s],
                   "uneven_shape": s, "uneven_neurons_per_layer": float(np.mean(kept[s])),
                   "uneven_total_parameters": routed + sum(counts[s]) * tc["hidden_size"]})
        (out / "config.json").write_text(json.dumps(c, indent=4))
        print(f"wrote {out}: {np.mean(kept[s]):,.0f} neurons in {np.mean(counts[s]):.0f} experts per layer; routed parameters "
              f"kept {routed / 1e9:.2f}B", flush=True)


if __name__ == "__main__":
    main()
