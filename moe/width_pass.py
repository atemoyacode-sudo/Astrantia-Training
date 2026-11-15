"""Same budget, different shape: is it better to keep fewer whole experts or more experts with fewer neurons each?

Every shape keeps the same number of routed-expert neurons per layer as the 48-expert cut (48 x 512 = 24,576), so total
parameters stay at ~8.5B:

  E experts x W neurons, top-K     48x512, 96x256, 128x192, 192x128, 256x96  (default K = 8)

Uneven widths (same neuron budget, experts of different widths; 2026-11-11):

  g128, g256f64, g128b28160        the budget's most important neurons among the top-E experts by routed mass, whatever
                                   expert they sit in (f = drop experts left with fewer neurons than this and hand
                                   their share to the others; b = another neuron budget per layer)
  t32x512+64x128                   tiers by routed mass: the top 32 experts at 512 neurons, the next 64 at 128

An uneven shape is run as a layer of full-width experts with the dropped neurons zeroed (the same function as ragged
experts; parameters are counted as kept neurons).

Experts are the E with the most routed mass on chat data; inside each expert the W neurons with the most importance on
the same chat data are kept (importance = routed weight x |silu(gate) * up| x norm of the neuron's down column, summed over
tokens). The router rows of the kept experts are kept unchanged. No training, no weight averaging.

One layer-by-layer pass over the original model. The original model's own stream over the chat calibration set gives the
statistics at each layer; every shape keeps its own stream over the held-out chat windows (A, B) and is scored at the end
by KL to the stored teacher. `full` (the original layer) is a sanity check: its KL should be ~0.

Inputs in --run: data.npz (C: calibration windows; A, B: held-out windows), teacher.npz (A/B top-64 log-probs), and
optionally wiki_router_stats.npz (keep_48 from the Wikipedia selection, scored as the baseline `wiki48x512`).

  python width_pass.py --source Qwen/Qwen3.5-35B-A3B --run /kaggle/working/run --dtype float32
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

from common import Source, build_config, ce_per_seq, kl_per_seq, layer_classes
from merge_pass import build_layer, build_layer_keys, closed_ci, run_layer

DEFAULT_SHAPES = ["48x512", "96x256", "128x192", "192x128", "256x96", "192x128k16"]


def parse_shape(s: str):
    """'192x128' or '192x128k16' -> (experts, width, top_k or None)."""
    e, rest = s.split("x")
    w, _, k = rest.partition("k")
    return int(e), int(w), (int(k) if k else None)


def parse_uneven(s: str):
    """'g128' / 'g256f64' / 'g128b28160' or 't32x512+64x128' -> a spec dict; None for an ExW shape."""
    if s.startswith("t"):
        return {"tiers": [tuple(int(v) for v in part.split("x")) for part in s[1:].split("+")]}
    if s.startswith("g"):
        body, _, b = s[1:].partition("b")
        e, _, f = body.partition("f")
        return {"experts": int(e), "floor": int(f) if f else 0, "budget": int(b) if b else None}
    return None


def uneven_keep(spec: dict, mass: np.ndarray, imp: np.ndarray, budget: int) -> np.ndarray:
    """The kept neurons of one layer, (E, I) bool, for an uneven shape. mass (E,), imp (E, I)."""
    order = np.argsort(-mass)
    keep = np.zeros(imp.shape, bool)
    if "tiers" in spec:
        rank = np.argsort(np.argsort(-imp, axis=1), axis=1)  # a neuron's rank inside its expert
        a = 0
        for e, w in spec["tiers"]:
            keep[order[a : a + e]] = rank[order[a : a + e]] < w
            a += e
        return keep
    ex = order[: spec["experts"]]
    sub = imp[ex].astype(np.float64).copy()
    n = min(spec["budget"] or budget, sub.size)
    for _ in range(64):
        k = sub >= np.sort(sub.ravel())[-n]
        w = k.sum(1)
        thin = (w > 0) & (w < spec["floor"])
        if not thin.any():
            break
        sub[thin] = -1.0  # this expert is dropped; its share goes to the others
    keep[ex] = k
    return keep


def masked_weights(sd: dict, keep: torch.Tensor):
    """Router rows and full-width expert weights of the experts with a kept neuron; dropped neurons are zeroed."""
    et = torch.where(keep.any(1))[0]
    k = keep.index_select(0, et)
    gu = sd["mlp.experts.gate_up_proj"].index_select(0, et)  # index_select already copies
    d = sd["mlp.experts.down_proj"].index_select(0, et)
    gu *= torch.cat([k, k], dim=1)[:, :, None].to(gu.dtype)
    d *= k[:, None, :].to(d.dtype)
    return {"mlp.gate.weight": sd["mlp.gate.weight"].index_select(0, et), "mlp.experts.gate_up_proj": gu, "mlp.experts.down_proj": d}


def chat_stats(x: torch.Tensor, gate, gu: torch.Tensor, d: torch.Tensor, chunk: int):
    """Routed mass per expert (E,) and neuron importance (E, I), from MoE inputs x (N, H) of the original layer."""
    n_exp, two_i, _ = gu.shape
    inter = two_i // 2
    acc = torch.float32 if x.device.type == "mps" else torch.float64  # MPS has no float64
    mass = torch.zeros(n_exp, dtype=acc, device=x.device)
    imp = torch.zeros(n_exp, inter, dtype=acc, device=x.device)
    for s in range(0, x.shape[0], chunk):
        xc = x[s : s + chunk]
        _, w, idx = gate(xc)
        mass.index_add_(0, idx.reshape(-1), w.reshape(-1).to(acc))
        for j in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == j)
            g, u = (xc[tok].float() @ gu[j].float().T).chunk(2, dim=-1)
            imp[j] += ((F.silu(g) * u).abs() * w[tok, slot, None].float()).sum(0).to(acc)
    imp *= d.float().norm(dim=1).to(acc)  # down_proj is (E, H, I): norm of each neuron's column
    return mass, imp


def shaped_weights(sd: dict, experts: np.ndarray, neurons: torch.Tensor | None):
    """Router rows and expert weights for the kept experts, each cut to the given neuron indices (rows of `neurons`)."""
    dev = sd["mlp.gate.weight"].device
    et = torch.as_tensor(np.sort(experts), device=dev)
    gu, d = sd["mlp.experts.gate_up_proj"].index_select(0, et), sd["mlp.experts.down_proj"].index_select(0, et)
    if neurons is not None:
        inter = d.shape[2]
        nb = neurons.index_select(0, et)  # (E', W)
        gate_rows = torch.gather(gu[:, :inter], 1, nb[:, :, None].expand(-1, -1, gu.shape[2]))
        up_rows = torch.gather(gu[:, inter:], 1, nb[:, :, None].expand(-1, -1, gu.shape[2]))
        gu = torch.cat([gate_rows, up_rows], dim=1)
        d = torch.gather(d, 2, nb[:, None, :].expand(-1, d.shape[1], -1))
    return {"mlp.gate.weight": sd["mlp.gate.weight"].index_select(0, et),
            "mlp.experts.gate_up_proj": gu.contiguous(), "mlp.experts.down_proj": d.contiguous()}


def free_cache(dev: str) -> None:
    """Give cached blocks back: uneven shapes change size every layer, so MPS never reuses them and fills the swap."""
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    elif dev.startswith("mps"):
        torch.mps.empty_cache()


def routed_rel_err(x, layer_a, layer_b) -> float:
    """||routed_a(x) - routed_b(x)|| / ||routed_a(x)|| for the MoE blocks (shared expert included in both)."""
    ya, yb = layer_a.mlp(x[None])[0], layer_b.mlp(x[None])[0]
    if isinstance(ya, tuple):
        ya, yb = ya[0], yb[0]
    return float((ya - yb).norm() / ya.norm().clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--shapes", nargs="*", default=DEFAULT_SHAPES)
    ap.add_argument("--budget-experts", type=int, default=48, help="budget = this many full-width experts per layer")
    ap.add_argument("--n-stats", type=int, default=100, help="calibration windows for the statistics; the rest of C "
                                                             "is used for the per-layer error diagnostic")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref", help="the variant the others are compared with (default: wiki48x512, else 48x512)")
    ap.add_argument("--out", default="width_results.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    run, dev, dt = Path(args.run), args.device, getattr(torch, args.dtype)
    src = Source(args.source)
    cfg = build_config(src.text_config)
    _, rotary_cls, norm_cls = layer_classes(cfg)
    data = dict(np.load(run / "data.npz"))
    teacher = dict(np.load(run / "teacher.npz"))
    wiki = run / "wiki_router_stats.npz"
    wiki_keep = np.load(wiki)["keep_48"] if wiki.exists() else None

    uneven = {s: parse_uneven(s) for s in args.shapes if parse_uneven(s)}
    shapes = {s: parse_shape(s) for s in args.shapes if s not in uneven}
    budget = args.budget_experts * cfg.moe_intermediate_size
    for s, (e, w, k) in shapes.items():
        if e > cfg.num_experts or w > cfg.moe_intermediate_size:
            raise SystemExit(f"shape {s} is larger than the model ({cfg.num_experts}x{cfg.moe_intermediate_size})")
        if e * w != budget:
            print(f"note: {s} keeps {e * w} neurons per layer, not {budget}", flush=True)
    names = ["full"] + list(args.shapes) + (["wiki48x512"] if wiki_keep is not None else [])
    cfgs = {}
    for s, (e, w, k) in shapes.items():
        c = copy.deepcopy(cfg)
        c.num_experts, c.moe_intermediate_size = e, w
        if k:
            c.num_experts_per_tok = k
        cfgs[s] = c
    if wiki_keep is not None:
        c = copy.deepcopy(cfg)
        c.num_experts = wiki_keep.shape[1]
        cfgs["wiki48x512"] = c

    t0 = time.time()
    embed = src.get("embed_tokens.weight").to(dev)
    rotary = rotary_cls(cfg).to(dev)
    calib = F.embedding(torch.as_tensor(data["C"], device=dev), embed).to(dt)
    held = {k: F.embedding(torch.as_tensor(data[k], device=dev), embed).to(dt) for k in ("A", "B")}
    streams = {v: {k: h.clone() for k, h in held.items()} for v in names}
    del held
    n_stats = min(args.n_stats, calib.shape[0])
    picks: dict[str, list] = {v: [] for v in names if v != "full"}
    diag: dict[str, list] = {v: [] for v in names if v != "full"}
    widths: dict[str, list] = {v: [] for v in uneven}  # per layer: kept neurons, kept experts, median width

    for i in range(cfg.num_hidden_layers):
        lt = time.time()
        sd = {k: src.get(f"layers.{i}.{k}").to(dev, dt) for k in build_layer_keys(cfg, i)}
        orig = build_layer(cfg, i, sd)
        calib, x = run_layer(orig, calib, rotary, args.microbatch, capture=True)
        x = x.reshape(calib.shape[0], calib.shape[1], -1)
        x_stats, x_diag = x[:n_stats].reshape(-1, x.shape[-1]), x[n_stats:].reshape(-1, x.shape[-1])
        del x
        mass, imp = chat_stats(x_stats, orig.mlp.gate, sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"], args.chunk)
        del x_stats
        order = torch.argsort(mass, descending=True).cpu().numpy()
        mass_np = imp_np = None
        for k in ("A", "B"):
            streams["full"][k], _ = run_layer(orig, streams["full"][k], rotary, args.microbatch)
        for v in names:
            if v == "full":
                continue
            student_sd = dict(sd)
            if v in uneven:
                if mass_np is None:
                    mass_np, imp_np = mass.cpu().numpy(), imp.cpu().numpy()
                keep = uneven_keep(uneven[v], mass_np, imp_np, budget)
                experts = np.where(keep.any(1))[0]
                student_sd.update(masked_weights(sd, torch.as_tensor(keep, device=dev)))
                c = copy.deepcopy(cfg)
                c.num_experts = len(experts)
                student = build_layer(c, i, student_sd)
                widths[v].append([int(keep.sum()), len(experts), int(np.median(keep.sum(1)[experts]))])
            else:
                if v == "wiki48x512":
                    experts, neurons = wiki_keep[i], None
                else:
                    e, w, _ = shapes[v]
                    experts = order[:e]
                    neurons = None if w == cfg.moe_intermediate_size else torch.sort(imp.topk(w, dim=1).indices, dim=1).values
                student_sd.update(shaped_weights(sd, experts, neurons))
                student = build_layer(cfgs[v], i, student_sd)
            if x_diag.shape[0]:
                diag[v].append(routed_rel_err(x_diag, orig, student))
            cover = float(mass[torch.as_tensor(np.asarray(experts), device=mass.device)].sum() / mass.sum())
            picks[v].append({"experts": np.sort(np.asarray(experts)).tolist(), "mass_covered": cover})
            for k in ("A", "B"):
                streams[v][k], _ = run_layer(student, streams[v][k], rotary, args.microbatch)
            del student, student_sd
            free_cache(dev)
        del orig, sd, x_diag
        free_cache(dev)
        msg = " ".join(f"{v}:{diag[v][-1]:.2f}" for v in diag if diag[v])
        print(f"layer {i + 1}/{cfg.num_hidden_layers} {time.time() - lt:.0f}s (total {time.time() - t0:.0f}s) "
              f"routed rel err: {msg}", flush=True)
    del calib

    norm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
    norm.load_state_dict({"weight": src.get("norm.weight").to(dev, dt)}, assign=True)
    lm_head = (embed if cfg.tie_word_embeddings else src.get("lm_head.weight").to(dev)).to(dt)
    del embed

    results = {"source": args.source, "dtype": args.dtype, "budget_neurons_per_layer": budget,
               "teacher_B_ce": float(teacher["B_ce"].mean()), "variants": {}}
    kls = {}
    for v in names:
        entry = {}
        per = []
        for k in ("A", "B"):
            kl, ce = [], []
            for s in range(streams[v][k].shape[0]):
                logits = F.linear(norm(streams[v][k][s : s + 1]), lm_head)
                kl.append(kl_per_seq(logits, torch.as_tensor(teacher[f"{k}_logp"][s : s + 1], device=dev),
                                     torch.as_tensor(teacher[f"{k}_idx"][s : s + 1], device=dev)).cpu())
                ce.append(ce_per_seq(logits, torch.as_tensor(data[k][s : s + 1], device=dev)).cpu())
            a = torch.cat(kl).numpy()
            per.append(a)
            entry[f"kl_{k}"], entry[f"ce_{k}"] = float(a.mean()), float(torch.cat(ce).mean())
        kls[v] = np.concatenate(per)
        entry["kl_AB"] = float(kls[v].mean())
        if v in diag and diag[v]:
            entry["routed_rel_err_mean"] = float(np.mean(diag[v]))
            entry["mass_covered_mean"] = float(np.mean([p["mass_covered"] for p in picks[v]]))
        results["variants"][v] = entry
    for v, w in widths.items():
        a = np.asarray(w)
        results["variants"][v].update(neurons_per_layer=float(a[:, 0].mean()), experts_per_layer=float(a[:, 1].mean()),
                                      median_width=float(a[:, 2].mean()))
    ref = args.ref or ("wiki48x512" if "wiki48x512" in kls else ("48x512" if "48x512" in kls else None))
    if ref:
        results["closed_vs"] = ref
        for v in names:
            if v not in (ref, "full"):
                results["variants"][v]["closed_AB"] = closed_ci(kls[v], kls[ref], args.boot, 0)
    print(f"\n{'shape':16s} {'KL A+B':>8s} {'closed vs ' + (ref or '-'):>24s} {'rel err':>8s} {'mass':>6s}")
    for v, e in results["variants"].items():
        c = e.get("closed_AB")
        cs = f"{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]" if c else "-"
        print(f"{v:16s} {e['kl_AB']:8.4f} {cs:>24s} {e.get('routed_rel_err_mean', float('nan')):8.3f} "
              f"{e.get('mass_covered_mean', float('nan')):6.3f}", flush=True)
    results["per_layer_rel_err"] = diag
    results["picks"] = picks
    out = Path(args.out)  # a bare file name goes into the run directory; a path is used as given
    out = run / out if out.parent == Path(".") else out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    np.savez(out.with_name(out.stem + "_kl.npz"), **kls)
    print("wrote", out, f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
