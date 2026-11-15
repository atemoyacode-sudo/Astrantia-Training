"""Uneven experts, "grouped" against "loop": same numbers, and how fast (one layer of the real size, random weights).

  python xmodel/moe/bench_uneven.py [--device cpu|cuda|mps] [--tokens 4096] [--dtype float32|float16] [--train-experts]

Checks: outputs agree, gradients (to the input, to the routing weights and, with --train-experts, to the expert
weights) agree with autograd through the loop. Then times forward + backward of each.
"""

from __future__ import annotations

import argparse
import time

import torch

import uneven


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=113)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--train-experts", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    g = torch.Generator().manual_seed(0)
    widths = torch.randint(1, 5, (args.experts,), generator=g) * 128  # 128..512 in steps of 128, like g128
    widths[0] = 1
    total = int(widths.sum())
    ex = uneven.RaggedExperts((torch.randn(2 * total, args.hidden, generator=g) * 0.02).to(dev, dt),
                              (torch.randn(args.hidden, total, generator=g) * 0.02).to(dev, dt), widths, torch.nn.SiLU())
    if args.train_experts:
        ex.gate_up.requires_grad_(True), ex.down.requires_grad_(True)
    x = torch.randn(args.tokens, args.hidden, generator=g).to(dev, dt).requires_grad_(True)
    logits = torch.randn(args.tokens, args.experts, generator=g).to(dev)
    wts, idx = torch.topk(torch.softmax(logits, -1), args.top_k, dim=-1)
    wts = wts.to(dt).requires_grad_(True)
    leaves = [x, wts] + ([ex.gate_up, ex.down] if args.train_experts else [])

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    def run(impl):
        ex.impl = impl
        out = ex(x, idx, wts)
        grads = torch.autograd.grad(out.float().square().sum(), leaves)
        return out, grads

    (oa, ga), (ob, gb) = run("loop"), run("grouped")
    rel = lambda a, b: float((a.detach().float() - b.detach().float()).abs().max() / (a.detach().float().abs().max() + 1e-12))  # noqa: E731
    print(f"{args.device} {args.dtype}: {args.tokens} tokens, {args.experts} experts, {total} neurons; "
          f"output rel diff {rel(oa, ob):.1e}; grad rel diffs {[f'{rel(a, b):.1e}' for a, b in zip(ga, gb)]}")
    tol = 2e-2 if dt == torch.float16 else 1e-4
    assert rel(oa, ob) < tol and all(rel(a, b) < tol for a, b in zip(ga, gb))
    for impl in ("loop", "grouped"):
        run(impl); sync()
        t = time.perf_counter()
        for _ in range(args.reps):
            run(impl)
        sync()
        dtm = (time.perf_counter() - t) / args.reps
        print(f"  {impl:8s} forward + backward {dtm * 1e3:8.1f} ms  ({args.tokens / dtm:,.0f} tokens/s for this one layer)")


if __name__ == "__main__":
    main()
