"""Experts of different widths in one layer (README 6h): storage and a forward pass that needs no padding.

A uniform checkpoint holds the routed experts as gate_up_proj (E, 2W, H) and down_proj (E, H, W). An uneven shape keeps
a different number of neurons in every expert, so here they are stored back to back:

  gate     (E, H)        router rows of the experts that keep a neuron
  gate_up  (2 * sum w, H)  expert e's rows start at 2 * offset[e]: its w[e] gate rows, then its w[e] up rows
  down     (H, sum w)      expert e's columns start at offset[e]
  widths   (E,)            w[e]

One file per layer (uneven_000.safetensors, ...) next to an ordinary checkpoint of everything else. The parameter count
is the kept neurons only; nothing is padded.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack(sd: dict, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """One layer's routed experts cut to `keep` ((E, I) bool), from the original layer's state dict."""
    gu, d = sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"]
    inter = d.shape[2]
    experts = torch.where(keep.any(1))[0]
    gate_up, down = [], []
    for e in experts.tolist():
        n = torch.where(keep[e])[0]
        gate_up += [gu[e, n], gu[e, inter + n]]
        down.append(d[e][:, n])
    return {"gate": sd["mlp.gate.weight"].index_select(0, experts).contiguous(), "gate_up": torch.cat(gate_up),
            "down": torch.cat(down, dim=1), "widths": keep.sum(1)[experts].to(torch.int64)}


class _RaggedUp(torch.autograd.Function):
    """x (S, H), sorted by expert, times each expert's own gate and up rows -> (S, 2 * wmax): the gate part in columns
    [0, w), the up part in [wmax, wmax + w), zeros elsewhere. One mm per part and expert on a contiguous slice, with a
    hand-written backward, so autograd keeps two nodes per layer instead of ten per expert."""

    @staticmethod
    def forward(ctx, x, gate_up, groups, wmax):
        ctx.groups, ctx.wmax = groups, wmax
        ctx.save_for_backward(x, gate_up)
        out = x.new_zeros(x.shape[0], 2 * wmax)
        for s, e, o, w in groups:
            torch.mm(x[s:e], gate_up[2 * o : 2 * o + w].T, out=out[s:e, :w])
            torch.mm(x[s:e], gate_up[2 * o + w : 2 * o + 2 * w].T, out=out[s:e, wmax : wmax + w])
        return out

    @staticmethod
    def backward(ctx, g):
        x, gate_up = ctx.saved_tensors
        wmax = ctx.wmax
        gx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(gate_up) if ctx.needs_input_grad[1] else None
        for s, e, o, w in ctx.groups:
            if gx is not None:
                torch.mm(g[s:e, :w], gate_up[2 * o : 2 * o + w], out=gx[s:e])
                gx[s:e].addmm_(g[s:e, wmax : wmax + w], gate_up[2 * o + w : 2 * o + 2 * w])
            if gw is not None:
                torch.mm(g[s:e, :w].T, x[s:e], out=gw[2 * o : 2 * o + w])
                torch.mm(g[s:e, wmax : wmax + w].T, x[s:e], out=gw[2 * o + w : 2 * o + 2 * w])
        return gx, gw, None, None


class _RaggedDown(torch.autograd.Function):
    """h (S, wmax), sorted by expert, times each expert's own down columns -> (S, H)."""

    @staticmethod
    def forward(ctx, h, down, groups):
        ctx.groups = groups
        ctx.save_for_backward(h, down)
        out = h.new_zeros(h.shape[0], down.shape[0])
        for s, e, o, w in groups:
            torch.mm(h[s:e, :w], down[:, o : o + w].T, out=out[s:e])
        return out

    @staticmethod
    def backward(ctx, g):
        h, down = ctx.saved_tensors
        gh = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        gd = torch.zeros_like(down) if ctx.needs_input_grad[1] else None
        for s, e, o, w in ctx.groups:
            if gh is not None:
                torch.mm(g[s:e], down[:, o : o + w], out=gh[s:e, :w])
            if gd is not None:
                torch.mm(g[s:e].T, h[s:e, :w], out=gd[:, o : o + w])
        return gh, gd, None


class _RaggedB(torch.autograd.Function):
    """z (S, r), sorted by expert, times each expert's own B[e] (H, r) -> (S, H) (README 26 B: the down_proj LoRA)."""

    @staticmethod
    def forward(ctx, z, b, egroups):
        ctx.egroups = egroups
        ctx.save_for_backward(z, b)
        out = z.new_zeros(z.shape[0], b.shape[1])
        for s, e, j in egroups:
            torch.mm(z[s:e], b[j].T, out=out[s:e])
        return out

    @staticmethod
    def backward(ctx, g):
        z, b = ctx.saved_tensors
        gz = torch.zeros_like(z) if ctx.needs_input_grad[0] else None
        gb = torch.zeros_like(b) if ctx.needs_input_grad[1] else None
        for s, e, j in ctx.egroups:
            if gz is not None:
                torch.mm(g[s:e], b[j], out=gz[s:e])
            if gb is not None:
                torch.mm(g[s:e].T, z[s:e], out=gb[j])
        return gz, gb, None


def add_down_lora(experts: "RaggedExperts", r: int, alpha: float) -> list[nn.Parameter]:
    """README 26 B: a rank-r update on every kept expert's down_proj, down_e += (alpha / r) * B[e] @ A_e.
    A is stored like `down` (r, sum w: expert e's columns at offset[e]), B as (E, H, r); B starts at zero, so the
    model is unchanged at the start. Router, widths and gate_up stay fixed. fold_down_lora() writes it into `down`."""
    dev = experts.down.device
    h = experts.down.shape[0]
    a = torch.cat([torch.randn(r, w, device=dev) / w ** 0.5 for w in experts.widths], dim=1)
    experts.lora_A = nn.Parameter(a)
    experts.lora_B = nn.Parameter(torch.zeros(experts.num_experts, h, r, device=dev))
    experts.lora_scale = alpha / r
    return [experts.lora_A, experts.lora_B]


def down_lora_delta(a: torch.Tensor, b: torch.Tensor, widths: list[int], scale: float) -> torch.Tensor:
    """The (H, sum w) change to `down` that a down_proj LoRA stands for."""
    out, o = [], 0
    for j, w in enumerate(widths):
        out.append(b[j].float() @ a[:, o : o + w].float() * scale)
        o += w
    return torch.cat(out, dim=1)


class RaggedExperts(nn.Module):
    """Drop-in for Qwen3_5MoeExperts on two flat tensors. `impl`: "grouped" (default) sorts the (token, expert) pairs
    by expert once and runs each expert on a contiguous slice, like transformers' grouped experts forward; "loop" is
    the plain loop over the experts that were picked (the first version; the two agree, see smoke_uneven.py)."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, widths: torch.Tensor, act_fn, impl: str = "grouped"):
        super().__init__()
        self.gate_up = nn.Parameter(gate_up, requires_grad=False)
        self.down = nn.Parameter(down, requires_grad=False)
        self.widths = [int(w) for w in widths]
        self.offsets = [0]
        for w in self.widths:
            self.offsets.append(self.offsets[-1] + w)
        self.num_experts = len(self.widths)
        self.wmax = max(self.widths)
        self.act_fn = act_fn
        self.impl = impl

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        return (self.forward_grouped if self.impl == "grouped" else self.forward_loop)(hidden_states, top_k_index, top_k_weights)

    def forward_grouped(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        tokens, k = top_k_index.shape
        ids, perm = torch.sort(top_k_index.reshape(-1))
        ends = torch.cumsum(torch.bincount(ids, minlength=self.num_experts), 0).tolist()  # the one GPU -> CPU sync
        groups, start = [], 0
        for e, end in enumerate(ends):
            if end > start:
                groups.append((start, end, self.offsets[e], self.widths[e]))
                start = end
        x = hidden_states[perm // k].to(self.gate_up.dtype)
        gate, up = _RaggedUp.apply(x, self.gate_up, groups, self.wmax).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        y = _RaggedDown.apply(h, self.down, groups)
        if getattr(self, "lora_A", None) is not None:  # README 26 B
            egroups, start = [], 0
            for e, end in enumerate(ends):
                if end > start:
                    egroups.append((start, end, e))
                    start = end
            z = _RaggedDown.apply(h, self.lora_A.to(h.dtype), groups)
            y = y + _RaggedB.apply(z, self.lora_B.to(h.dtype), egroups) * self.lora_scale
        y = y * top_k_weights.reshape(-1)[perm, None]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return y[inv].view(tokens, k, -1).sum(dim=1).to(hidden_states.dtype)

    def forward_loop(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            o, w = self.offsets[e], self.widths[e]
            top_k_pos, token_idx = torch.where(mask[e])
            gate, up = F.linear(hidden_states[token_idx], self.gate_up[2 * o : 2 * o + 2 * w]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            y = F.linear(h, self.down[:, o : o + w])
            if getattr(self, "lora_A", None) is not None:  # README 26 B
                y = y + F.linear(F.linear(h, self.lora_A[:, o : o + w].to(h.dtype)), self.lora_B[e].to(h.dtype)) * self.lora_scale
            out.index_add_(0, token_idx, (y * top_k_weights[token_idx, top_k_pos, None]).to(out.dtype))
        return out


def attach_layer(layer: nn.Module, t: dict[str, torch.Tensor]) -> None:
    """Replace one decoder layer's router rows and routed experts by an uneven set (tensors as written by pack)."""
    ref = layer.mlp.shared_expert_gate.weight
    layer.mlp.gate.weight = nn.Parameter(t["gate"].to(ref.device, ref.dtype), requires_grad=False)
    layer.mlp.gate.num_experts = t["gate"].shape[0]
    layer.mlp.experts = RaggedExperts(t["gate_up"].to(ref.device, ref.dtype), t["down"].to(ref.device, ref.dtype),
                                      t["widths"], layer.mlp.experts.act_fn)


def attach(model: nn.Module, path: str | Path) -> int:
    """Give every layer of a loaded model its uneven experts from `path`; returns the routed neurons kept in total."""
    from safetensors.torch import load_file

    kept = 0
    for i, layer in enumerate(model.model.layers):
        t = load_file(Path(path) / f"uneven_{i:03d}.safetensors")
        attach_layer(layer, t)
        kept += int(t["widths"].sum())
    return kept
