"""Fold a heal_train.py LoRA into an uneven student checkpoint, in place (README 14: the released weights).

The routed experts (uneven_*.safetensors) are not LoRA targets and stay as they are. The attention, Gated DeltaNet and
shared-expert linears get W += scale * B @ A with scale = alpha / r = 2, as in heal_train.add_lora and
pad_uneven.fold_lora. Works shard by shard on the CPU (no model load), so it needs little memory.

  python fold_lora_uneven.py --student /tmp/student --lora lora_final.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="the uneven checkpoint directory; its shards are rewritten")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="alpha / r (heal_train uses alpha = 2r)")
    args = ap.parse_args()
    d = Path(args.student)
    st = torch.load(args.lora, map_location="cpu", weights_only=False)
    st = st.get("lora", st)
    mods = sorted(k[:-2] for k in st if k.endswith(".A"))
    assert len(mods) * 2 == len(st) and all(m + ".B" in st for m in mods), "unexpected LoRA keys"
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    by_file, key_of = defaultdict(list), {}
    for m in mods:  # "model.layers.0.linear_attn.out_proj" -> the checkpoint key (it may carry a language_model prefix)
        suffix = "." + m[len("model."):] + ".weight"
        hits = [k for k in wmap if k == m + ".weight" or k.endswith(suffix)]
        assert len(hits) == 1, f"{m}: {len(hits)} checkpoint keys match"
        key_of[m] = hits[0]
        assert not wmap[hits[0]].startswith("uneven_"), f"{m} sits in an expert file"
        by_file[wmap[hits[0]]].append(m)
    rel = []
    for fname, ms in sorted(by_file.items()):
        t = load_file(d / fname)
        for m in ms:
            w = t[key_of[m]]
            delta = (st[m + ".B"].float() @ st[m + ".A"].float()) * args.scale
            assert delta.shape == w.shape, f"{m}: LoRA {tuple(delta.shape)} vs weight {tuple(w.shape)}"
            new = (w.float() + delta).to(w.dtype)
            rel.append(float((new.float() - w.float()).norm() / w.float().norm().clamp_min(1e-12)))
            t[key_of[m]] = new.contiguous()
        save_file(t, d / fname, metadata={"format": "pt"})
        print(f"{fname}: folded {len(ms)}", flush=True)
    out = {"lora": Path(args.lora).name, "folded": len(mods), "files": len(by_file), "scale": args.scale,
           "relative_change_mean": sum(rel) / len(rel), "relative_change_max": max(rel)}
    (d / "fold_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
