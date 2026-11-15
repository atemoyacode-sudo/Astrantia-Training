"""Build servable checkpoints for one or more shapes from width_pass.py (E experts x W neurons, picked on chat data).

One pass over the original model's own stream on the calibration windows (C) gives, per layer, the routed mass per
expert and the importance per neuron; each shape is cut from that and saved as an ordinary Qwen3_5MoeForCausalLM
checkpoint (bf16) with the tokenizer and chat template, so vLLM can serve it. The statistics are saved too
(stats.npz: mass (L, E), imp (L, E, I)), so more shapes can be cut later with --from-stats and no GPU pass.

Uneven shapes (g128, g256f64, t32x512+64x128, ...; README 6h) are saved as an ordinary checkpoint of everything but the
routed experts (placeholder experts in the config, "uneven_shape" set) plus uneven_NNN.safetensors per layer
(uneven.py); heal_train.py loads them. They cannot be served by vLLM as they are.

  python build_shape.py --source Qwen/Qwen3.5-35B-A3B --run /kaggle/working/run --shapes 96x256 48x512 --out /tmp/ckpt
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import Source, build_config, layer_classes
from merge_pass import build_layer, build_layer_keys, finish_variant, run_layer, save_layer
import uneven
from safetensors.torch import save_file
from width_pass import chat_stats, parse_shape, parse_uneven, shaped_weights, uneven_keep


def copy_tokenizer(repo: str, out: Path) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(repo).save_pretrained(out)
    for f in ("chat_template.jinja", "generation_config.json"):
        try:
            shutil.copy(hf_hub_download(repo, f), out / f)
        except Exception as e:  # the tokenizer usually carries the template already
            print(f"  (no {f}: {e})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--run", required=True, help="has data.npz (C = calibration windows)")
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="one sub-directory per shape")
    ap.add_argument("--from-stats", help="stats.npz from an earlier run: skip the GPU pass")
    ap.add_argument("--budget-experts", type=int, default=48, help="uneven shapes keep this many full-width experts' "
                                                                   "worth of neurons per layer (unless the shape says bN)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--save-dtype", default="bfloat16")
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    run, out, dev, dt, sdt = Path(args.run), Path(args.out), args.device, getattr(torch, args.dtype), getattr(torch, args.save_dtype)
    src = Source(args.source)
    cfg = build_config(src.text_config)
    rotary_cls = layer_classes(cfg)[1]
    ragged = {s: parse_uneven(s) for s in args.shapes if parse_uneven(s)}
    shapes = {s: parse_shape(s) for s in args.shapes if s not in ragged}
    expert_keys = ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    kept = {s: [] for s in ragged}
    cfgs = {}
    for s, (e, w, k) in shapes.items():
        c = copy.deepcopy(cfg)
        c.num_experts, c.moe_intermediate_size = e, w
        if k:
            c.num_experts_per_tok = k
        cfgs[s] = c
    stats = dict(np.load(args.from_stats)) if args.from_stats else {"mass": [], "imp": []}
    weight_maps = {s: {} for s in args.shapes}

    t0 = time.time()
    embed = src.get("embed_tokens.weight")
    rotary = rotary_cls(cfg).to(dev)
    calib = None
    if not args.from_stats:
        calib = torch.nn.functional.embedding(torch.as_tensor(np.load(run / "data.npz")["C"], device=dev), embed.to(dev)).to(dt)
    # read the next layer's weights (CPU) while this one is computed: over HTTP, reading is the slow part
    pool = ThreadPoolExecutor(1)
    fetch = lambda i: {k: src.get(f"layers.{i}.{k}") for k in build_layer_keys(cfg, i)}
    nxt = pool.submit(fetch, 0)
    for i in range(cfg.num_hidden_layers):
        lt = time.time()
        cpu_sd = nxt.result()
        if i + 1 < cfg.num_hidden_layers:
            nxt = pool.submit(fetch, i + 1)
        if args.from_stats:
            sd = cpu_sd
            mass, imp = torch.as_tensor(stats["mass"][i]), torch.as_tensor(stats["imp"][i])
        else:
            sd = {k: t.to(dev, dt) for k, t in cpu_sd.items()}
            orig = build_layer(cfg, i, sd)
            calib, x = run_layer(orig, calib, rotary, args.microbatch, capture=True)
            mass, imp = chat_stats(x, orig.mlp.gate, sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"], args.chunk)
            del x, orig
            stats["mass"].append(mass.cpu().numpy())
            stats["imp"].append(imp.cpu().numpy().astype(np.float32))
        order = torch.argsort(mass, descending=True).cpu().numpy()
        for s, (e, w, _) in shapes.items():
            neurons = None if w == cfg.moe_intermediate_size else torch.sort(imp.topk(w, dim=1).indices, dim=1).values.to(sd["mlp.gate.weight"].device)
            layer_sd = dict(sd)
            layer_sd.update(shaped_weights(sd, order[:e], neurons))
            save_layer(out / s, i, {f"model.layers.{i}.{k}": t.to(sdt).contiguous().cpu() for k, t in layer_sd.items()},
                       weight_maps[s])
        for s, spec in ragged.items():
            keep = torch.as_tensor(uneven_keep(spec, mass.cpu().numpy(), imp.cpu().numpy(),
                                               args.budget_experts * cfg.moe_intermediate_size))
            t = uneven.pack({k: sd[k].cpu() for k in expert_keys}, keep)
            (out / s).mkdir(parents=True, exist_ok=True)
            save_file({k: (v if k == "widths" else v.to(sdt)).contiguous() for k, v in t.items()}, out / s / f"uneven_{i:03d}.safetensors")
            kept[s].append([int(keep.sum()), int(keep.any(1).sum())])
            save_layer(out / s, i, {f"model.layers.{i}.{k}": v.to(sdt).contiguous().cpu() for k, v in sd.items()
                                    if k not in expert_keys}, weight_maps[s])
        del sd, cpu_sd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"layer {i + 1}/{cfg.num_hidden_layers} {time.time() - lt:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
    if not args.from_stats:
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "stats.npz", mass=np.stack(stats["mass"]), imp=np.stack(stats["imp"]))

    rest = {"model.embed_tokens.weight": embed.to(sdt), "model.norm.weight": src.get("norm.weight").to(sdt)}
    if not cfg.tie_word_embeddings:
        rest["lm_head.weight"] = src.get("lm_head.weight").to(sdt)
    for s, (e, w, k) in shapes.items():
        finish_variant(out / s, weight_maps[s], rest, src.text_config, e)
        conf = json.loads((out / s / "config.json").read_text())
        conf["moe_intermediate_size"] = w
        if k:
            conf["num_experts_per_tok"] = k
        conf["torch_dtype"] = args.save_dtype
        (out / s / "config.json").write_text(json.dumps(conf, indent=1))
        copy_tokenizer(args.source if not Path(args.source).exists() else "Qwen/Qwen3.5-35B-A3B", out / s)
        print(f"saved {out / s}", flush=True)
    for s in ragged:
        finish_variant(out / s, weight_maps[s], rest, src.text_config, cfg.num_experts_per_tok)
        conf = json.loads((out / s / "config.json").read_text())
        n = np.array(kept[s])
        # placeholder experts (num_experts_per_tok of them, 8 neurons wide): uneven.attach() puts the real ones in
        conf.update({"moe_intermediate_size": 8, "torch_dtype": args.save_dtype, "uneven_shape": s,
                     "uneven_neurons_per_layer": float(n[:, 0].mean()), "uneven_experts_per_layer": float(n[:, 1].mean())})
        (out / s / "config.json").write_text(json.dumps(conf, indent=1))
        copy_tokenizer(args.source if not Path(args.source).exists() else "Qwen/Qwen3.5-35B-A3B", out / s)
        print(f"saved {out / s}: {n[:, 0].mean():,.0f} neurons in {n[:, 1].mean():.0f} experts per layer", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
