"""Rehearsal 3: how much of the pruning loss does LoRA training win back?

Loads the pruned checkpoint (run/pruned_M), adds LoRA, and trains it to match the teacher's top-k next-token
distribution on the training set T (built by `prune_pass.py --extra-train`). Measures KL on A every --eval-every steps,
keeps the LoRA with the lowest A, and reports KL / CE on B with `closed = 1 - KL / KL(step 0)`.

  lin      LoRA on every nn.Linear in the decoder layers (attention, GDN, shared expert)
  lin_exp  lin, plus a per-expert LoRA on the kept routed experts' gate_up_proj and down_proj

  python lora_train.py --run /content/runs/q35a3b --keep 48 --variant lin_exp
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LayerState, ce_per_seq, kl_per_seq


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


class ExpertLoRA(nn.Module):
    """Per-expert LoRA on a Qwen3.5-MoE experts module (gate_up_proj (E, 2I, H), down_proj (E, H, I))."""

    def __init__(self, experts: nn.Module, r: int, alpha: float):
        super().__init__()
        e, two_i, h = experts.gate_up_proj.shape
        inter = two_i // 2
        dev = experts.gate_up_proj.device
        self.scale = alpha / r
        self.A1 = nn.Parameter(torch.randn(e, r, h, device=dev) / math.sqrt(h))
        self.B1 = nn.Parameter(torch.zeros(e, two_i, r, device=dev))
        self.A2 = nn.Parameter(torch.randn(e, r, inter, device=dev) / math.sqrt(inter))
        self.B2 = nn.Parameter(torch.zeros(e, h, r, device=dev))
        self.experts = [experts]  # not a submodule: its weights stay where they are
        experts.forward = self.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Sort the (token, slot) pairs by expert once (one host sync instead of one per expert), split the LoRA
        factors once (one stacked gradient instead of a full-size one per expert), and do the LoRA maths in the
        activations' dtype."""
        ex = self.experts[0]
        dt = hidden_states.dtype
        out = torch.zeros_like(hidden_states)
        k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=ex.gate_up_proj.shape[0]).tolist()
        tok_all, w_all = order // k, top_k_weights.reshape(-1)[order]
        a1, b1, a2, b2 = (t.to(dt).unbind(0) for t in (self.A1, self.B1, self.A2, self.B2))
        start = 0
        for j, c in enumerate(counts):
            if c == 0:
                continue
            tok, w = tok_all[start : start + c], w_all[start : start + c]
            start += c
            x = hidden_states[tok]
            gu = F.linear(x, ex.gate_up_proj[j]) + (x @ a1[j].T) @ b1[j].T * self.scale
            g, u = gu.chunk(2, dim=-1)
            h = ex.act_fn(g) * u
            y = F.linear(h, ex.down_proj[j]) + (h @ a2[j].T) @ b2[j].T * self.scale
            out.index_add_(0, tok, (y * w[:, None]).to(out.dtype))
        return out

    def forward_reference(self, hidden_states, top_k_index, top_k_weights):
        """The first, slow version (one torch.where per expert, float32 LoRA maths). Kept to test the fast one."""
        ex = self.experts[0]
        out = torch.zeros_like(hidden_states)
        for j in range(ex.gate_up_proj.shape[0]):
            tok, slot = torch.where(top_k_index == j)
            if len(tok) == 0:
                continue
            x = hidden_states[tok]
            gu = F.linear(x, ex.gate_up_proj[j]) + ((x.float() @ self.A1[j].T) @ self.B1[j].T * self.scale).to(x.dtype)
            g, u = gu.chunk(2, dim=-1)
            h = ex.act_fn(g) * u
            y = F.linear(h, ex.down_proj[j]) + ((h.float() @ self.A2[j].T) @ self.B2[j].T * self.scale).to(h.dtype)
            out.index_add_(0, tok, (y * top_k_weights[tok, slot, None]).to(out.dtype))
        return out


class RouterDelta(nn.Module):
    """A trainable float32 change to a Qwen3.5-MoE router's weight (zero at the start), same softmax / top-k /
    renormalise as the original router."""

    def __init__(self, gate: nn.Module):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(gate.weight.shape, dtype=torch.float32, device=gate.weight.device))
        self.gate = [gate]
        gate.forward = self.forward

    def forward(self, hidden_states):
        g = self.gate[0]
        h = hidden_states.reshape(-1, g.hidden_dim)
        logits = F.linear(h, g.weight).float() + h.float() @ self.delta.T
        probs = torch.softmax(logits, dim=-1)
        top, idx = torch.topk(probs, g.top_k, dim=-1)
        top = top / top.sum(dim=-1, keepdim=True)
        return logits.to(h.dtype), top.to(h.dtype), idx


def add_lora(model, variant: str, r_lin: int, r_exp: int, router: bool = False) -> list[nn.Parameter]:
    holders, routers = nn.ModuleList(), nn.ModuleList()
    for layer in model.model.layers:
        for name, mod in list(layer.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and child.out_features > 1:
                    setattr(mod, cname, LoRALinear(child, r_lin, 2 * r_lin))
        if variant == "lin_exp":
            holders.append(ExpertLoRA(layer.mlp.experts, r_exp, 2 * r_exp))
        if router:
            routers.append(RouterDelta(layer.mlp.gate))
    model.expert_lora = holders  # registers the expert LoRA parameters on the model
    if router:
        model.router_delta = routers
    return [p for n, p in model.named_parameters() if p.requires_grad]


def lora_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def evaluate(model, ids, t_logp, t_idx, dev, microbatch, with_ce=False):
    model.eval()
    kls, ces = [], []
    with torch.no_grad():
        for s in range(0, len(ids), microbatch):
            x = torch.as_tensor(ids[s : s + microbatch], device=dev)
            logits = model(input_ids=x, use_cache=False).logits
            kls.append(kl_per_seq(logits, torch.as_tensor(t_logp[s : s + microbatch], device=dev),
                                  torch.as_tensor(t_idx[s : s + microbatch], device=dev)).cpu())
            if with_ce:
                ces.append(ce_per_seq(logits, x).cpu())
    return torch.cat(kls).numpy(), (torch.cat(ces).numpy() if with_ce else None)


def closed_ci(kl_m, kl_none, n_boot, seed):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        r = rng.integers(0, len(kl_m), len(kl_m))
        boots.append(1 - kl_m[r].mean() / kl_none[r].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(1 - kl_m.mean() / kl_none.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--keep", type=int, default=48)
    ap.add_argument("--teacher", default="teacher")
    ap.add_argument("--variant", choices=["lin", "lin_exp"], default="lin_exp")
    ap.add_argument("--r-lin", type=int, default=16)
    ap.add_argument("--r-exp", type=int, default=8)
    ap.add_argument("--router", action="store_true", help="also train a change to every router's weight")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--eval-microbatch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-n", type=int, default=0, help="use only the first N sequences of T (0 = all)")
    ap.add_argument("--tag", default="", help="suffix for the output names, to keep runs with different settings apart")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="stop after N steps, as if the VM were lost (for testing)")
    ap.add_argument("--force", action="store_true", help="run even if the outputs already exist")
    args = ap.parse_args()

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(args.seed)
    run, dev = Path(args.run), args.device
    data = dict(np.load(run / "data.npz"))
    teacher = dict(np.load(run / f"{args.teacher}.npz"))
    assert "T" in data and "T_logp" in teacher, "run prune_pass.py --extra-train first"
    name = f"lora_{args.keep}_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    out = run / f"{name}.json"
    if args.train_n:
        data["T"], teacher["T_logp"], teacher["T_idx"] = data["T"][: args.train_n], teacher["T_logp"][: args.train_n], teacher["T_idx"][: args.train_n]
    if out.exists() and not args.force:
        print(f"already done ({out}); pass --force to run again", flush=True)
        return
    sig = {k: v for k, v in vars(args).items() if k not in ("stop_after_steps", "force", "device", "boot")}
    state = LayerState(run / f"_state_{name}.pt", sig)
    saved_state = state.load()

    model = Qwen3_5MoeForCausalLM.from_pretrained(run / f"pruned_{args.keep}", dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(dev).requires_grad_(False)
    params = add_lora(model, args.variant, args.r_lin, args.r_exp, args.router)
    n_params = sum(p.numel() for p in params)
    print(f"{args.variant}: {n_params / 1e6:.1f}M trainable parameters", flush=True)

    ev = lambda k, ce=False: evaluate(model, data[k], teacher[f"{k}_logp"], teacher[f"{k}_idx"], dev, args.eval_microbatch, ce)
    if saved_state:
        kl_a0, kl_b0, ce_b0 = saved_state["kl_a0"], saved_state["kl_b0"], saved_state["ce_b0"]
    else:
        kl_a0, _ = ev("A")
        kl_b0, ce_b0 = ev("B", True)
    print(f"step 0: KL_A {kl_a0.mean():.4f} KL_B {kl_b0.mean():.4f} CE_B {ce_b0.mean():.4f}", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_t = len(data["T"])
    steps = int(args.epochs * n_t / args.microbatch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = np.random.default_rng(args.seed)
    order = np.concatenate([rng.permutation(n_t) for _ in range(math.ceil(args.epochs) + 1)])
    t_ids, t_logp, t_idx = data["T"], teacher["T_logp"], teacher["T_idx"]

    best = (float(kl_a0.mean()), 0, lora_state(model))
    curve = [{"step": 0, "kl_A": float(kl_a0.mean())}]
    t0, losses, start = time.time(), [], 1
    if saved_state:
        named = dict(model.named_parameters())
        for n, v in saved_state["lora"].items():
            named[n].data.copy_(v.to(dev))
        opt.load_state_dict(saved_state["opt"])
        sched.load_state_dict(saved_state["sched"])
        best, curve, losses, start = saved_state["best"], saved_state["curve"], saved_state["losses"], saved_state["step"] + 1
        print(f"resuming at step {start}", flush=True)
    for step in range(start, steps + 1):
        model.train()
        rows = order[(step - 1) * args.microbatch : step * args.microbatch]
        x = torch.as_tensor(t_ids[rows], device=dev)
        logits = model(input_ids=x, use_cache=False).logits
        loss = kl_per_seq(logits, torch.as_tensor(t_logp[rows], device=dev), torch.as_tensor(t_idx[rows], device=dev)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if step % args.eval_every == 0 or step == steps:
            kl_a, _ = ev("A")
            curve.append({"step": step, "kl_A": float(kl_a.mean()), "train_kl": float(np.mean(losses[-args.eval_every:]))})
            if kl_a.mean() < best[0]:
                best = (float(kl_a.mean()), step, lora_state(model))
            print(f"step {step}/{steps}: train KL {curve[-1]['train_kl']:.4f}  KL_A {kl_a.mean():.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if step < steps:
                state.save({"step": step, "lora": lora_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "best": best, "curve": curve, "losses": losses, "kl_a0": kl_a0, "kl_b0": kl_b0, "ce_b0": ce_b0})
        if args.stop_after_steps and step >= args.stop_after_steps:
            raise SystemExit(f"stopping after step {step} (--stop-after-steps)")

    named = dict(model.named_parameters())
    for n, v in best[2].items():
        named[n].data.copy_(v.to(dev))
    kl_a, _ = ev("A")
    kl_b, ce_b = ev("B", True)
    result = {"variant": args.variant, "tag": args.tag, "keep": args.keep, "r_lin": args.r_lin, "r_exp": args.r_exp, "router": args.router,
              "train_n": len(t_ids), "epochs": args.epochs, "trainable_params": n_params, "steps": steps,
              "train_tokens": int(steps * args.microbatch * t_ids.shape[1]), "lr": args.lr, "best_step": best[1],
              "step0": {"kl_A": float(kl_a0.mean()), "kl_B": float(kl_b0.mean()), "ce_B": float(ce_b0.mean())},
              "best": {"kl_A": float(kl_a.mean()), "kl_B": float(kl_b.mean()), "ce_B": float(ce_b.mean())},
              "closed_A": closed_ci(kl_a, kl_a0, args.boot, args.seed), "closed_B": closed_ci(kl_b, kl_b0, args.boot, args.seed),
              "teacher_B_ce": float(teacher["B_ce"].mean()), "curve": curve, "seconds": round(time.time() - t0)}
    c = result["closed_B"]
    result["H4"] = "recovers" if c[0] >= 0.25 and c[1] > 0 else ("no" if c[0] < 0.10 else "partly")
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}), flush=True)
    torch.save(best[2], run / f"{name}.pt")
    out.write_text(json.dumps(result, indent=1))
    state.clear()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
