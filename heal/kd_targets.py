"""README 21: the original's top-k next-token log-probabilities on every position of the KD windows, with vLLM.

vLLM's prompt_logprobs[t] is the distribution of token t given the tokens before it, so the target for the student's
logits at position p (which predict token p + 1) is prompt_logprobs[p + 1]: positions 0 .. len-2 get targets.

  python kd_targets.py --model Qwen/Qwen3.5-35B-A3B --windows kd_windows.npz --k 20 --out kd_targets.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = np.load(args.windows)["ids"]
    n, length = ids.shape
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=length + 16, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=False, max_logprobs=args.k, max_num_seqs=args.batch)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=args.k, temperature=0.0)
    logp = np.zeros((n, length - 1, args.k), dtype=np.float16)
    idx = np.zeros((n, length - 1, args.k), dtype=np.int32)
    t0 = time.time()
    for s in range(0, n, args.batch):
        outs = llm.generate([TokensPrompt(prompt_token_ids=row.tolist()) for row in ids[s : s + args.batch]], sp,
                            use_tqdm=False)
        for j, o in enumerate(outs):
            pl = o.prompt_logprobs  # [None, {tok: Logprob}, ...], one per prompt position
            for p in range(length - 1):
                entries = sorted(pl[p + 1].items(), key=lambda kv: kv[1].rank)[: args.k]
                entries = [e for e in entries if e[1].rank <= args.k]  # the actual token may sit outside the top-k
                for r, (t, lp) in enumerate(entries):
                    idx[s + j, p, r], logp[s + j, p, r] = t, lp.logprob
                for r in range(len(entries), args.k):  # (never for vLLM's top-k, kept for safety) an empty slot
                    idx[s + j, p, r], logp[s + j, p, r] = 0, -1e4
        print(f"{min(s + args.batch, n)}/{n} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(args.out, logp=logp, idx=idx)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
