"""KL to the stored teacher for an MLX model (e.g. a 4-bit conversion from to_mlx.py), on the held-out windows A and B.

Compare with width_pass.py's KL for the same shape in float32 to see what the conversion and quantization cost.

  python mlx_kl.py --run ../../results/xmodel_q35a3b/chat_run --model ../cache/mlx_96x256_q4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import kl_per_seq


def main() -> None:
    import mlx.core as mx
    from mlx_lm import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="an mlx-lm adapter directory (heal/to_mlx_adapter.py)")
    ap.add_argument("--out", default=None, help="result file (default: kl_chat.json in the adapter or model directory; "
                                                "give one when an adapter is measured on a model it was not made for)")
    args = ap.parse_args()
    run = Path(args.run)
    data, teacher = np.load(run / "data.npz"), np.load(run / "teacher.npz")
    model, _ = load(args.model, adapter_path=args.adapter)
    kls = []
    for k in ("A", "B"):
        for s in range(len(data[k])):
            logits = model(mx.array(data[k][s : s + 1])).astype(mx.float32)
            lt = torch.from_numpy(np.array(logits))
            kls.append(float(kl_per_seq(lt, torch.as_tensor(teacher[f"{k}_logp"][s : s + 1]),
                                        torch.as_tensor(teacher[f"{k}_idx"][s : s + 1]))[0]))
    res = {"model": args.model, "adapter": args.adapter, "kl_AB": float(np.mean(kls)), "kl_AB_per_seq": kls}
    out = Path(args.out) if args.out else Path(args.adapter or args.model) / "kl_chat.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"KL A+B {res['kl_AB']:.4f} over {len(kls)} windows -> {out}", flush=True)


if __name__ == "__main__":
    main()
