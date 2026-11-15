"""Rehearsal 2: merge the removed experts into the kept ones, in one layer-by-layer pass over the original MoE.

Uses the kept experts and their assignment from phase 1 (`router_stats.npz`), the data (`data.npz`) and the teacher
(`teacher.npz`) in --run. Every variant keeps its own hidden-state stream, so each layer is built from the variant's own
inputs (not the original model's). At the end, every variant's KL to the teacher on A and B is computed from the
streams, and every variant is saved as an ordinary checkpoint with M experts.

Variants (the router rows are never changed):
  none       cut only
  avg        average the assigned experts' gate/up/down, weighted by the routed mass they had on C
  align      the same, after matching each removed expert's intermediate neurons to the kept expert's (Hungarian)
  lsq        keep gate/up, refit all M down_proj at once by ridge regression so that the cut layer's routed-expert
             output matches the original layer's on C (shrunk towards the current down_proj); ridge strength picked
             per layer on A, with "no refit" as one of the choices
  align_lsq  lsq on top of align

  python merge_pass.py --source Qwen/Qwen3.5-35B-A3B --run /content/runs/q35a3b --keep 48
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
from safetensors.torch import save_file
from scipy.optimize import linear_sum_assignment

from common import LayerState, Source, build_config, ce_per_seq, kl_per_seq, layer_classes, positions, to_cpu, to_dev

VARIANTS = ["none", "avg", "align", "lsq", "align_lsq"]
EXPERT_KEYS = ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def closed_ci(kl_m, kl_none, n_boot, seed):
    rng = np.random.default_rng(seed)
    point = 1 - kl_m.mean() / kl_none.mean()
    boots = []
    for _ in range(n_boot):
        r = rng.integers(0, len(kl_m), len(kl_m))
        boots.append(1 - kl_m[r].mean() / kl_none[r].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------- building a variant's expert weights


def neuron_signature(gu: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(I, 3H): for each intermediate neuron, its gate row, up row and down column, normalised."""
    inter = d.shape[1]
    sig = torch.cat([gu[:inter], gu[inter:], d.T], dim=1).float()
    return F.normalize(sig, dim=1)


def align_to(gu_k, d_k, gu_e, d_e):
    """Permute expert e's intermediate neurons to best match expert k's."""
    inter = d_k.shape[1]
    sim = neuron_signature(gu_k, d_k) @ neuron_signature(gu_e, d_e).T
    _, perm = linear_sum_assignment(-sim.cpu().numpy())
    perm = torch.as_tensor(perm, device=gu_e.device)
    gu_p = torch.cat([gu_e[:inter][perm], gu_e[inter:][perm]], dim=0)
    return gu_p, d_e[:, perm]


def merged_experts(sd, keep, assign, mass, align: bool):
    """Mass-weighted average of each kept expert with the experts assigned to it."""
    gu_all, d_all = sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"]
    gu_out = torch.empty((len(keep),) + gu_all.shape[1:], dtype=torch.float32, device=gu_all.device)
    d_out = torch.empty((len(keep),) + d_all.shape[1:], dtype=torch.float32, device=d_all.device)
    m = np.maximum(mass, 1e-12)
    for j, k in enumerate(keep):
        members = np.nonzero(assign == k)[0]
        wsum = m[members].sum()
        gu_acc = gu_all[k].float() * (m[k] / wsum)
        d_acc = d_all[k].float() * (m[k] / wsum)
        for e in members:
            if e == k:
                continue
            gu_e, d_e = gu_all[e], d_all[e]
            if align:
                gu_e, d_e = align_to(gu_all[k], d_all[k], gu_e, d_e)
            gu_acc += gu_e.float() * (m[e] / wsum)
            d_acc += d_e.float() * (m[e] / wsum)
        gu_out[j], d_out[j] = gu_acc, d_acc
    return gu_out.to(gu_all.dtype), d_out.to(d_all.dtype)


def start_weights(variant, sd, keep, assign, mass):
    kt = torch.as_tensor(keep, device=sd["mlp.gate.weight"].device)
    out = {"mlp.gate.weight": sd["mlp.gate.weight"].index_select(0, kt)}
    if variant in ("avg", "align", "align_lsq"):
        gu, d = merged_experts(sd, keep, assign, mass, align=variant != "avg")
    else:
        gu = sd["mlp.experts.gate_up_proj"].index_select(0, kt)
        d = sd["mlp.experts.down_proj"].index_select(0, kt)
    out["mlp.experts.gate_up_proj"], out["mlp.experts.down_proj"] = gu.contiguous(), d.contiguous()
    return out


def build_layer(cfg, idx, sd):
    layer_cls, _, _ = layer_classes(cfg)
    with torch.device("meta"):
        layer = layer_cls(cfg, idx)
    layer.load_state_dict(sd, assign=True, strict=True)
    return layer.eval()


# ---------------------------------------------------------------- running layers and capturing the MoE input


def run_layer(layer, hidden, rotary, microbatch, capture=False):
    """Run `layer` over `hidden`; with capture=True also return the MoE block's inputs, flattened to (N, H)."""
    out = torch.empty_like(hidden)
    xs = []
    hook = layer.mlp.register_forward_pre_hook(lambda _m, inp: xs.append(inp[0].reshape(-1, inp[0].shape[-1]))) if capture else None
    for s in range(0, hidden.shape[0], microbatch):
        h = hidden[s : s + microbatch]
        mpos, tpos = positions(h.shape[0], h.shape[1], h.device)
        with torch.no_grad():
            out[s : s + microbatch] = layer(h, position_embeddings=rotary(h, mpos), attention_mask=None, position_ids=tpos)
    if hook:
        hook.remove()
    return out, (torch.cat(xs) if capture else None)


def routed_features(x, gate, gu, chunk):
    """Yield (rows, phi) per chunk: phi[n, j*I:(j+1)*I] = p_j(x_n) * act(gate_j x_n) * (up_j x_n) for the experts the
    router picks (zero elsewhere), in float32. The routed-expert output is phi @ D with D[j*I+i, :] = down_j[:, i]."""
    n_exp, two_i, _ = gu.shape
    inter = two_i // 2
    for s in range(0, x.shape[0], chunk):
        xc = x[s : s + chunk]
        with torch.no_grad():
            _, w, idx = gate(xc)
        phi = torch.zeros(xc.shape[0], n_exp * inter, dtype=torch.float32, device=x.device)
        xf = xc.float()
        for j in range(n_exp):
            tok, slot = torch.where(idx == j)
            if len(tok) == 0:
                continue
            g, u = (xf[tok] @ gu[j].float().T).chunk(2, dim=-1)
            phi[tok, j * inter : (j + 1) * inter] = F.silu(g) * u * w[tok, slot, None].float()
        yield slice(s, s + xc.shape[0]), phi


def flat_down(d):
    """(E, H, I) -> (E*I, H) in float32, matching routed_features' layout."""
    return d.float().permute(0, 2, 1).reshape(-1, d.shape[1])


def routed_output(x, gate, gu, d, chunk):
    dflat = flat_down(d)
    out = torch.empty(x.shape[0], d.shape[1], dtype=torch.float32, device=x.device)
    for rows, phi in routed_features(x, gate, gu, chunk):
        out[rows] = phi @ dflat
    return out


def lsq_refit(student, orig, x_c, x_a, alphas, chunk):
    """Ridge refit of all down_proj of `student` (in place). Returns a dict of diagnostics."""
    gu, d = student.mlp.experts.gate_up_proj, student.mlp.experts.down_proj
    dflat = flat_down(d)
    feat = dflat.shape[0]
    gram = torch.zeros(feat, feat, dtype=torch.float32, device=x_c.device)
    rhs = torch.zeros(feat, d.shape[1], dtype=torch.float32, device=x_c.device)
    target_c = routed_output(x_c, orig.mlp.gate, orig.mlp.experts.gate_up_proj, orig.mlp.experts.down_proj, chunk)
    err_c = 0.0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows, phi in routed_features(x_c, student.mlp.gate, gu, chunk):
        resid = target_c[rows] - phi @ dflat
        err_c += float(resid.pow(2).sum())
        gram += phi.T @ phi
        rhs += phi.T @ resid
    del target_c
    target_a = routed_output(x_a, orig.mlp.gate, orig.mlp.experts.gate_up_proj, orig.mlp.experts.down_proj, chunk)
    phi_a = torch.cat([phi for _, phi in routed_features(x_a, student.mlp.gate, gu, chunk)])
    resid_a = target_a - phi_a @ dflat
    del target_a
    base_a = float(resid_a.pow(2).sum())

    g64 = gram.double()
    del gram
    scale = float(g64.diagonal().mean())
    best = (base_a, None, None)
    tried = {}
    for alpha in alphas:
        a = g64.clone()
        a.diagonal().add_(alpha * scale)
        chol, info = torch.linalg.cholesky_ex(a)
        del a
        if int(info) != 0:
            tried[alpha] = None
            continue
        delta = torch.cholesky_solve(rhs.double(), chol).float()
        del chol
        err = float((resid_a - phi_a @ delta).pow(2).sum())
        tried[alpha] = err / max(base_a, 1e-30)
        if err < best[0]:
            best = (err, alpha, delta)
        else:
            del delta
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    err_a, alpha, delta = best
    if delta is not None:
        n_exp, hid, inter = d.shape
        new = (dflat + delta).reshape(n_exp, inter, hid).permute(0, 2, 1)
        d.data.copy_(new.to(d.dtype))
    return {"alpha": alpha, "rel_err_A": err_a / max(base_a, 1e-30), "tried": tried,
            "base_err_A_per_tok": base_a / x_a.shape[0], "base_err_C_per_tok": err_c / x_c.shape[0]}


# ---------------------------------------------------------------- saving


def save_layer(dirpath: Path, i: int, tensors: dict, weight_map: dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    fname = f"layer_{i:03d}.safetensors"
    save_file(tensors, dirpath / fname)
    weight_map.update({k: fname for k in tensors})


def finish_variant(dirpath: Path, weight_map: dict, rest: dict, text_config: dict, m: int):
    save_file(rest, dirpath / "rest.safetensors")
    wm = {**weight_map, **{k: "rest.safetensors" for k in rest}}
    (dirpath / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
    tc = dict(text_config)
    tc.update({"num_experts": m, "model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"]})
    (dirpath / "config.json").write_text(json.dumps(tc, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--keep", type=int, default=48)
    ap.add_argument("--teacher", default="teacher")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--alphas", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--logit-microbatch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=4096, help="tokens per chunk when building the regression")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10, help="save a resumable state every N layers (0 = never)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N layers, as if the VM were lost (for testing)")
    ap.add_argument("--force", action="store_true", help="run even if the outputs already exist")
    args = ap.parse_args()
    torch.set_grad_enabled(False)  # the layers' weights are Parameters; nothing here needs autograd

    run, dev, m = Path(args.run), args.device, args.keep
    src = Source(args.source)
    cfg = build_config(src.text_config)
    cfg_m = copy.deepcopy(cfg)
    cfg_m.num_experts = m
    _, rotary_cls, norm_cls = layer_classes(cfg)
    data = dict(np.load(run / "data.npz"))
    teacher = dict(np.load(run / f"{args.teacher}.npz"))
    st = np.load(run / "router_stats.npz")
    keep_all, assign_all, mass_all = st[f"keep_{m}"], st[f"assign_{m}"], st["mass"]
    out = run / f"merge_{m}_{args.teacher}.json"
    if out.exists() and not args.force and (args.no_save or all((run / f"merged_{v}_{m}" / "config.json").exists() for v in args.variants)):
        print(f"already done ({out}); pass --force to run again", flush=True)
        return

    t0 = time.time()
    embed = src.get("embed_tokens.weight").to(dev)
    rotary = rotary_cls(cfg).to(dev)
    variants = args.variants
    sets = ("C", "A", "B")
    h0 = {k: F.embedding(torch.as_tensor(data[k], device=dev), embed) for k in sets}
    hidden = {v: {k: h0[k].clone() for k in sets} for v in variants}
    del h0
    weight_maps: dict[str, dict] = {v: {} for v in variants}
    diag: dict[str, list] = {v: [] for v in variants if v.endswith("lsq")}
    state = LayerState(run / f"_state_merge_{m}_{args.teacher}.pt",
                       {"source": args.source, "keep": m, "variants": variants, "alphas": args.alphas, "save": not args.no_save})
    start = 0
    saved_state = state.load()
    if saved_state:
        start, hidden = saved_state["next_layer"], to_dev(saved_state["hidden"], dev)
        weight_maps, diag = saved_state["weight_maps"], saved_state["diag"]
        missing = [f"merged_{v}_{m}/{f}" for v in variants for f in set(weight_maps[v].values())
                   if not (run / f"merged_{v}_{m}" / f).exists()]
        if missing:
            raise SystemExit(f"resume state says layers < {start} are saved, but these files are missing: {missing[:3]}")
        print(f"resuming at layer {start + 1}", flush=True)

    for i in range(start, cfg.num_hidden_layers):
        lt = time.time()
        sd = {k: src.get(f"layers.{i}.{k}").to(dev) for k in build_layer_keys(cfg, i)}
        orig = build_layer(cfg, i, sd)
        keep, assign, mass = keep_all[i], assign_all[i], mass_all[i]
        for v in variants:
            student_sd = dict(sd)
            student_sd.update(start_weights(v, sd, keep, assign, mass))
            student = build_layer(cfg_m, i, student_sd)
            if v.endswith("lsq"):
                _, x_c = run_layer(student, hidden[v]["C"], rotary, args.microbatch, capture=True)
                _, x_a = run_layer(student, hidden[v]["A"], rotary, args.microbatch, capture=True)
                info = lsq_refit(student, orig, x_c, x_a, args.alphas, args.chunk)
                del x_c, x_a
                diag[v].append(info)
            for k in sets:
                hidden[v][k], _ = run_layer(student, hidden[v][k], rotary, args.microbatch)
            if not args.no_save:
                save_layer(run / f"merged_{v}_{m}", i, {f"model.layers.{i}.{k}": t.detach().contiguous().cpu()
                                                        for k, t in student.state_dict().items()}, weight_maps[v])
            del student, student_sd
        del orig, sd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        extra = " ".join(f"{v}:a={diag[v][-1]['alpha']},err={diag[v][-1]['rel_err_A']:.3f}" for v in diag)
        print(f"layer {i + 1}/{cfg.num_hidden_layers} done, {time.time() - lt:.0f}s (total {time.time() - t0:.0f}s) {extra}", flush=True)
        if args.ckpt_every and (i + 1) % args.ckpt_every == 0 and i + 1 < cfg.num_hidden_layers:
            state.save({"next_layer": i + 1, "hidden": to_cpu(hidden), "weight_maps": weight_maps, "diag": diag})
            print(f"  saved resume state at layer {i + 1}", flush=True)
        if args.stop_after and i + 1 >= args.stop_after:
            raise SystemExit(f"stopping after layer {i + 1} (--stop-after)")

    norm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
    norm.load_state_dict({"weight": src.get("norm.weight").to(dev)}, assign=True)
    lm_head = embed if cfg.tie_word_embeddings else src.get("lm_head.weight").to(dev)

    results = {"keep": m, "teacher": args.teacher, "teacher_B_ce": float(teacher["B_ce"].mean()), "variants": {}}
    kls = {}
    for v in variants:
        entry = {}
        for k in ("A", "B"):
            kl, ce = [], []
            for s in range(0, hidden[v][k].shape[0], args.logit_microbatch):
                sl = slice(s, s + args.logit_microbatch)
                with torch.no_grad():
                    logits = F.linear(norm(hidden[v][k][sl]), lm_head)
                    kl.append(kl_per_seq(logits, torch.as_tensor(teacher[f"{k}_logp"][sl], device=dev),
                                         torch.as_tensor(teacher[f"{k}_idx"][sl], device=dev)).cpu())
                    ce.append(ce_per_seq(logits, torch.as_tensor(data[k][sl], device=dev)).cpu())
            kls[(v, k)] = torch.cat(kl).numpy()
            entry[f"kl_{k}"] = float(kls[(v, k)].mean())
            entry[f"ce_{k}"] = float(torch.cat(ce).mean())
        results["variants"][v] = entry
    if "none" in variants:
        for v in variants:
            if v != "none":
                for k in ("A", "B"):
                    results["variants"][v][f"closed_{k}"] = closed_ci(kls[(v, k)], kls[("none", k)], args.boot, args.seed)
        merged = [v for v in variants if v != "none"]
        if merged:
            pick = min(merged, key=lambda v: results["variants"][v]["kl_A"])
            c = results["variants"][pick]["closed_B"]
            results["picked_by_A"] = pick
            results["H3"] = bool(c[0] >= 0.10 and c[1] > 0)
    for v, e in results["variants"].items():
        print(v, json.dumps(e), flush=True)
    if "H3" in results:
        print(f"picked by A: {results['picked_by_A']} | H3 (closed_B >= 0.10, CI low > 0): {results['H3']}", flush=True)
    results["lsq_layers"] = diag
    out = run / f"merge_{m}_{args.teacher}.json"
    out.write_text(json.dumps(results, indent=1))
    np.savez(run / f"merge_{m}_{args.teacher}_kl.npz", **{f"{v}_{k}": a for (v, k), a in kls.items()})
    print("wrote", out, flush=True)

    if not args.no_save:
        rest = {"model.embed_tokens.weight": embed.cpu(), "model.norm.weight": norm.weight.detach().cpu()}
        if not cfg.tie_word_embeddings:
            rest["lm_head.weight"] = lm_head.cpu()
        for v in variants:
            finish_variant(run / f"merged_{v}_{m}", weight_maps[v], rest, src.text_config, m)
            print(f"saved {run / f'merged_{v}_{m}'}", flush=True)
    state.clear()
    print(f"total {time.time() - t0:.0f}s")


def build_layer_keys(cfg, idx):
    layer_cls, _, _ = layer_classes(cfg)
    with torch.device("meta"):
        return list(layer_cls(cfg, idx).state_dict().keys())


if __name__ == "__main__":
    main()
