"""README 21: token-level KD. The cut student (uneven, bf16) with a LoRA (started from a given LoRA, e.g. wikil-a's)
is trained so that its next-token distribution matches the original's top-k log-probabilities on every position
of the KD windows (kd_windows.py + kd_targets.py): KL over the top-k plus one bucket for the rest, as kl_per_seq.

Held-out KL (top-64 targets) on the chat windows A, B and, if given, the general windows H, at the start and every
--eval-every steps; the LoRA with the lowest chat KL is saved as lora_best.pt, the last one as lora_final.pt.

  python kd_train.py --student /content/student/g128b28160 --init-lora lora_wikila.pt --windows kd_windows.npz \\
      --targets kd_targets.npz --run heal_inputs --general general_run --out /content/kd21
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "moe"))


def kd_chunk(h, w, t_lp, t_ix):
    """KL(teacher || student) summed over the positions of a chunk: top-k + one bucket for the rest."""
    logq = torch.log_softmax((h @ w.T).float(), dim=-1)
    q_top = logq.gather(-1, t_ix.long())
    p_top = t_lp.float().exp()
    p_tail = (1 - p_top.sum(-1)).clamp_min(1e-8)
    q_tail = (1 - q_top.exp().sum(-1)).clamp_min(1e-8)
    return ((p_top * (t_lp.float() - q_top)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())).sum()


class SharedExtra(torch.nn.Module):
    """README 23: extra neurons for a layer's shared expert (as if shared_expert_intermediate_size were larger):
    a SwiGLU block added to the shared expert's output, so the shared expert's own gate still scales it.
    gate / up start like the model's own (normal, std 0.02), down at zero: the model starts unchanged."""

    def __init__(self, shared, width: int):
        super().__init__()
        base = shared.gate_proj.base if hasattr(shared.gate_proj, "base") else shared.gate_proj
        h, dev = base.in_features, base.weight.device
        self.gate = torch.nn.Parameter(torch.randn(width, h, device=dev) * 0.02)
        self.up = torch.nn.Parameter(torch.randn(width, h, device=dev) * 0.02)
        self.down = torch.nn.Parameter(torch.zeros(h, width, device=dev))
        self.orig = [shared.forward]
        shared.forward = self.forward

    def forward(self, x):
        dt = x.dtype
        g, u = x @ self.gate.to(dt).T, x @ self.up.to(dt).T
        return self.orig[0](x) + (torch.nn.functional.silu(g) * u) @ self.down.to(dt).T


def window_kd(model, ids, t_lp, t_ix, chunk: int):
    """Summed KL over positions 0 .. len-2 of one window (batch 1)."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0][:-1]
    w = model.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        total = total + checkpoint(kd_chunk, h[s:e], w, t_lp[s:e], t_ix[s:e], use_reentrant=False)
    return total, h.shape[0]


@torch.no_grad()
def held_out_kl(model, data, teacher, parts, dev) -> float:
    from common import kl_per_seq

    model.eval()
    kls = []
    for k in parts:
        for s in range(len(data[k])):
            x = torch.as_tensor(data[k][s : s + 1], device=dev)
            logits = model(input_ids=x, use_cache=False).logits
            logits = logits[:, : teacher[f"{k}_logp"].shape[1]]  # kd_targets.py output has len - 1 positions (README 27)
            kls.append(float(kl_per_seq(logits, torch.as_tensor(teacher[f"{k}_logp"][s : s + 1], device=dev),
                                        torch.as_tensor(teacher[f"{k}_idx"][s : s + 1], device=dev))[0]))
    model.train()
    return float(np.mean(kls))


def main() -> None:
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    from heal_train import add_lora, attach_uneven, gdn_in_float32, lora_state

    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--init-lora", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--run", required=True, help="data.npz + teacher.npz (chat A, B)")
    ap.add_argument("--general", default=None, help="data.npz (H) + teacher_H.npz (general held-out)")
    ap.add_argument("--ja-eval", nargs=2, default=None, metavar=("WINDOWS", "TARGETS"),
                    help="README 27: held-out Japanese windows (ja_held.py) and their kd_targets.py output")
    ap.add_argument("--expert-r", type=int, default=0, help="README 26 B: rank of a LoRA on every kept expert's down_proj")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8, help="windows per optimizer step")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--keep-every-eval", action="store_true", help="also save the LoRA at every evaluation")
    ap.add_argument("--loss-chunk", type=int, default=256)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--shared-extra", type=int, default=0, help="README 23: extra shared-expert neurons per layer")
    ap.add_argument("--extra-lr", type=float, default=1e-4)
    ap.add_argument("--eval-only", action="store_true", help="README 22: measure the start (step 0) and stop")
    ap.add_argument("--smoke", type=int, default=0, help="stop after N steps (speed / memory check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    dev = "cuda"
    torch.manual_seed(0)
    gdn_in_float32()
    cfg = AutoConfig.from_pretrained(args.student)
    model = Qwen3_5MoeForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, device_map=dev,
                                                  attn_implementation="sdpa")
    attach_uneven(model, cfg, args.student)
    model.requires_grad_(False)
    params = add_lora(model, args.r, 2 * args.r)
    saved = torch.load(args.init_lora, map_location="cpu", weights_only=False)
    saved = saved.get("lora", saved)
    expert_r = args.expert_r or next((v.shape[0] for k, v in saved.items() if k.endswith("experts.lora_A")), 0)
    if expert_r:  # README 26 B: added after the plain LoRA, so a plain start LoRA loads as it is
        import uneven

        for layer in model.model.layers:
            params += uneven.add_down_lora(layer.mlp.experts, expert_r, 2 * expert_r)
        n_ex = sum(l.mlp.experts.lora_A.numel() + l.mlp.experts.lora_B.numel() for l in model.model.layers)
        log(f"down_proj LoRA rank {expert_r} on every kept expert: {n_ex / 1e6:.1f}M parameters")
    named = dict(model.named_parameters())
    if expert_r and not any(k.endswith("experts.lora_A") for k in saved):  # a plain start: the expert part starts fresh
        saved = {**saved, **{k: v.detach().cpu().clone() for k, v in named.items() if "experts.lora_" in k}}
    assert set(saved) == {k for k, v in named.items() if v.requires_grad}, "the LoRA does not match the model"
    for k, v in saved.items():
        named[k].data.copy_(v.to(named[k].device))
    extra = []
    if args.shared_extra:
        model.shared_extra = torch.nn.ModuleList(SharedExtra(layer.mlp.shared_expert, args.shared_extra)
                                                 for layer in model.model.layers)
        extra = list(model.shared_extra.parameters())
        log(f"shared expert +{args.shared_extra} neurons per layer: {sum(p.numel() for p in extra) / 1e6:.1f}M parameters")
    log(f"started from {args.init_lora}: {len(saved)} tensors, {sum(p.numel() for p in params) / 1e6:.1f}M parameters")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    run = Path(args.run)
    chat_data, chat_t = dict(np.load(run / "data.npz")), dict(np.load(run / "teacher.npz"))
    gen = None
    if args.general:
        g = Path(args.general)
        gen = (dict(np.load(g / "data.npz")), dict(np.load(g / "teacher_H.npz")))

    ja = None
    if args.ja_eval:
        jt = np.load(args.ja_eval[1])
        ja = ({"J": np.load(args.ja_eval[0])["ids"]}, {"J_logp": jt["logp"], "J_idx": jt["idx"]})

    def evaluate(step):
        r = {"step": step, "kl_chat": held_out_kl(model, chat_data, chat_t, ("A", "B"), dev)}
        if gen:
            r["kl_general"] = held_out_kl(model, gen[0], gen[1], ("H",), dev)
        if ja:
            r["kl_japanese"] = held_out_kl(model, ja[0], ja[1], ("J",), dev)
        return r

    wins = np.load(args.windows)
    ids_all, src = wins["ids"], wins["src"]
    tg = np.load(args.targets)
    t_lp_all, t_ix_all = tg["logp"], tg["idx"]
    n = len(ids_all)
    steps = int(math.ceil(n * args.epochs / args.batch))
    groups = [{"params": params, "lr": args.lr}] + ([{"params": extra, "lr": args.extra_lr}] if extra else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    params = params + extra
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.1, 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    results = [evaluate(0)]
    log(json.dumps(results[-1]))
    if args.eval_only:
        (out / "results.json").write_text(json.dumps({"results": results, "args": vars(args)}, indent=1))
        log("EVAL_DONE")
        return
    best = (results[-1]["kl_chat"], 0)
    torch.save(lora_state(model), out / "lora_best.pt")
    t0 = time.time()
    order = np.random.default_rng(0).permutation(int(n * args.epochs + args.batch)) % n
    pos = 0
    for step in range(1, steps + 1):
        tot, cnt = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for _ in range(args.batch):
            i = int(order[pos])
            pos += 1
            ids = torch.as_tensor(ids_all[i : i + 1].astype(np.int64), device=dev)
            lp = torch.as_tensor(t_lp_all[i], device=dev)
            ix = torch.as_tensor(t_ix_all[i], device=dev)
            loss, m = window_kd(model, ids, lp, ix, args.loss_chunk)
            (loss / (m * args.batch)).backward()
            tot, cnt = tot + float(loss), cnt + m
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == 1:
            log(f"step {step}/{steps} kd {tot / cnt:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if args.smoke and step >= args.smoke:
            log("smoke run: stopping")
            break
        if step % args.eval_every == 0 or step == steps or (time.time() - t0) / 3600 > args.max_hours:
            results.append(evaluate(step))
            log(json.dumps(results[-1]))
            if args.keep_every_eval:  # README 24: a checkpoint may meet both KL rules before the end (11-18)
                torch.save(lora_state(model), out / f"lora_step{step:04d}.pt")
            if results[-1]["kl_chat"] < best[0]:
                best = (results[-1]["kl_chat"], step)
                torch.save(lora_state(model), out / "lora_best.pt")
            (out / "results.json").write_text(json.dumps({"results": results, "best_step": best[1], "args": vars(args)}, indent=1))
            if (time.time() - t0) / 3600 > args.max_hours:
                log("out of time")
                break
    torch.save(lora_state(model), out / "lora_final.pt")
    (out / "results.json").write_text(json.dumps({"results": results, "best_step": best[1], "args": vars(args)}, indent=1))
    log(f"KD_DONE best chat KL {best[0]:.4f} at step {best[1]}")


if __name__ == "__main__":
    main()
