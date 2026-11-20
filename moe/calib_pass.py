"""Which windows choose the neurons? (README 6l) One pass over the original model, on the Mac.

The current cut kept the neurons that matter on chat windows of code and math. Here the same shape is chosen three
ways - from those chat statistics (a stored stats.npz = the current model), from statistics on general-knowledge
windows, and from the two added after scaling each to sum 1 per layer - and each is scored by KL to the original
model's own output on held-out chat windows (A, B) and held-out general windows (H).

  python moe/calib_pass.py --source Qwen/Qwen3.5-35B-A3B --run cache/chat_run --general cache/general/data_general.npz \\
         --chat-stats cache/shape/stats.npz --shape g128b28160 --out cache/general/calib

Writes <out>_results.json, <out>_stats_general.npz and <out>_stats_mix.npz (mass (L, E), imp (L, E, I)).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import Source, build_config, kl_per_seq, layer_classes
from merge_pass import build_layer, build_layer_keys, run_layer
from width_pass import chat_stats, free_cache, masked_weights, parse_uneven, uneven_keep


def full_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over the whole vocabulary, mean over positions, per sequence."""
    lp, lq = F.log_softmax(teacher_logits.float(), -1), F.log_softmax(student_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--run", required=True, help="directory with data.npz (A, B) and teacher.npz")
    ap.add_argument("--general", required=True, help="npz with G (calibration) and H (held-out) general windows")
    ap.add_argument("--chat-stats", required=True)
    ap.add_argument("--shape", default="g128b28160")
    ap.add_argument("--budget-experts", type=int, default=48)
    ap.add_argument("--mix-weights", type=float, nargs="*", default=[0.5], help="share of the general statistics in each mix")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--layers", type=int, help="stop after this many layers (a dry run)")
    ap.add_argument("--extra-held", nargs="*", default=[], metavar="NAME=NPZ:KEY",
                    help="README 30: more held-out windows scored the same way (e.g. Hg=cache/general/data_general.npz:H)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    run, dev, dt = Path(args.run), args.device, getattr(torch, args.dtype)
    src = Source(args.source)
    cfg = build_config(src.text_config)
    _, rotary_cls, norm_cls = layer_classes(cfg)
    data, teacher, gen = dict(np.load(run / "data.npz")), dict(np.load(run / "teacher.npz")), dict(np.load(args.general))
    chat = np.load(args.chat_stats)
    spec, budget = parse_uneven(args.shape), args.budget_experts * cfg.moe_intermediate_size
    mixes = {f"mix{int(round(w * 100)):02d}": w for w in args.mix_weights}
    names = ["full", "chat", "general"] + list(mixes)
    held_ids = {"A": data["A"], "B": data["B"], "H": gen["H"]}
    for extra in args.extra_held:
        name, rest = extra.split("=", 1)
        path, key = rest.rsplit(":", 1)
        held_ids[name] = np.load(path)[key]

    t0 = time.time()
    embed = src.get("embed_tokens.weight").to(dev)
    rotary = rotary_cls(cfg).to(dev)
    calib = F.embedding(torch.as_tensor(gen["G"], device=dev), embed).to(dt)
    held = torch.cat([F.embedding(torch.as_tensor(v, device=dev), embed).to(dt) for v in held_ids.values()])
    streams = {v: held.clone() for v in names}
    del held
    n_layers = args.layers or cfg.num_hidden_layers
    g_mass, g_imp, cover, overlap = [], [], [], {v: [] for v in names if v not in ("full", "chat")}
    for i in range(n_layers):
        lt = time.time()
        sd = {k: src.get(f"layers.{i}.{k}").to(dev, dt) for k in build_layer_keys(cfg, i)}
        orig = build_layer(cfg, i, sd)
        calib, x = run_layer(orig, calib, rotary, args.microbatch, capture=True)
        mass, imp = chat_stats(x, orig.mlp.gate, sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"], args.chunk)
        del x
        mg, ig = mass.cpu().numpy().astype(np.float64), imp.cpu().numpy().astype(np.float64)
        mc, ic = chat["mass"][i].astype(np.float64), chat["imp"][i].astype(np.float64)
        g_mass.append(mg.astype(np.float32)), g_imp.append(ig.astype(np.float32))
        stats = {"chat": (mc, ic), "general": (mg, ig)}
        for v, w in mixes.items():
            stats[v] = ((1 - w) * mc / mc.sum() + w * mg / mg.sum(), (1 - w) * ic / ic.sum() + w * ig / ig.sum())
        keeps = {v: uneven_keep(spec, m, im, budget) for v, (m, im) in stats.items()}
        kc = keeps["chat"]
        # how much of each kind of text's activity sits in what the current model kept
        cover.append([float(ic[kc].sum() / ic.sum()), float(ig[kc].sum() / ig.sum()),
                      float(mc[kc.any(1)].sum() / mc.sum()), float(mg[kc.any(1)].sum() / mg.sum())])
        for v in overlap:
            overlap[v].append(float((keeps[v] & kc).sum() / kc.sum()))
        streams["full"], _ = run_layer(orig, streams["full"], rotary, args.microbatch)
        for v, keep in keeps.items():
            student_sd = dict(sd)
            student_sd.update(masked_weights(sd, torch.as_tensor(keep, device=dev)))
            c = copy.deepcopy(cfg)
            c.num_experts = int(keep.any(1).sum())
            student = build_layer(c, i, student_sd)
            streams[v], _ = run_layer(student, streams[v], rotary, args.microbatch)
            del student, student_sd
            free_cache(dev)
        del orig, sd
        free_cache(dev)
        print(f"layer {i + 1}/{n_layers} {time.time() - lt:.0f}s (total {time.time() - t0:.0f}s) kept by the current model: "
              f"chat importance {cover[-1][0]:.2f}, general importance {cover[-1][1]:.2f}; "
              + " ".join(f"{v} shares {overlap[v][-1]:.2f} of its neurons" for v in overlap), flush=True)
    del calib

    norm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
    norm.load_state_dict({"weight": src.get("norm.weight").to(dev, dt)}, assign=True)
    lm_head = (embed if cfg.tie_word_embeddings else src.get("lm_head.weight").to(dev)).to(dt)
    del embed
    spans, a = {}, 0
    for k, v in held_ids.items():
        spans[k] = (a, a + v.shape[0])
        a += v.shape[0]
    cov = np.asarray(cover)
    res = {"source": args.source, "shape": args.shape, "layers": n_layers, "windows": {"G": int(gen["G"].shape[0]), **{k: int(v.shape[0]) for k, v in held_ids.items()}},
           "kept_by_current": {"chat_importance": float(cov[:, 0].mean()), "general_importance": float(cov[:, 1].mean()),
                               "chat_mass": float(cov[:, 2].mean()), "general_mass": float(cov[:, 3].mean()), "per_layer": cov.tolist()},
           "neurons_shared_with_current": {v: float(np.mean(o)) for v, o in overlap.items()}, "variants": {}}
    for v in names:
        e = {}
        for k, (s0, s1) in spans.items():
            kls, top = [], []
            for s in range(s0, s1):
                logits = F.linear(norm(streams[v][s : s + 1]), lm_head)
                kls.append(float(full_kl(logits, F.linear(norm(streams["full"][s : s + 1]), lm_head))))
                if k in ("A", "B"):
                    j = s - s0
                    top.append(float(kl_per_seq(logits, torch.as_tensor(teacher[f"{k}_logp"][j : j + 1], device=dev),
                                                torch.as_tensor(teacher[f"{k}_idx"][j : j + 1], device=dev))))
            e[f"kl_{k}"] = float(np.mean(kls))
            if top:
                e[f"kl_{k}_stored_teacher"] = float(np.mean(top))
        e["kl_chat"] = (e["kl_A"] + e["kl_B"]) / 2
        e["kl_chat_stored_teacher"] = (e["kl_A_stored_teacher"] + e["kl_B_stored_teacher"]) / 2
        e["kl_general"] = e["kl_H"]
        res["variants"][v] = e
        print(f"{v:10s} KL chat {e['kl_chat']:.4f} (stored teacher {e['kl_chat_stored_teacher']:.4f})  KL general {e['kl_general']:.4f}", flush=True)
    Path(f"{args.out}_results.json").write_text(json.dumps(res, indent=1))
    if n_layers == cfg.num_hidden_layers:
        gm, gi = np.stack(g_mass), np.stack(g_imp)
        np.savez(f"{args.out}_stats_general.npz", mass=gm, imp=gi)
        for v, w in mixes.items():
            mc, ic = chat["mass"].astype(np.float64), chat["imp"].astype(np.float64)
            m = (1 - w) * mc / mc.sum(1, keepdims=True) + w * gm / gm.sum(1, keepdims=True)
            im = (1 - w) * ic / ic.sum((1, 2), keepdims=True) + w * gi / gi.sum((1, 2), keepdims=True)
            np.savez(f"{args.out}_stats_{v}.npz", mass=m.astype(np.float32), imp=im.astype(np.float32))
    print(f"done in {time.time() - t0:.0f}s -> {args.out}_results.json")


if __name__ == "__main__":
    main()
