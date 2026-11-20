"""Fold a heal_train.py LoRA into an uneven student checkpoint, in place (README 14: the released weights).

The routed experts (uneven_*.safetensors) stay as they are, unless the LoRA carries README 26 B's down_proj update
(keys ...mlp.experts.lora_A / lora_B): then each layer's `down` gets uneven.down_lora_delta. The attention, Gated DeltaNet and
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
    expert = {k: st.pop(k) for k in [k for k in st if ".mlp.experts.lora_" in k]}
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
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "moe"))
    from uneven import down_lora_delta

    ex_rel = []
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in expert})
    for i in layers:
        pre = next(k for k in expert if f".layers.{i}.mlp.experts.lora_A" in k)[: -len("lora_A")]
        a, b = expert[pre + "lora_A"], expert[pre + "lora_B"]
        f = d / f"uneven_{i:03d}.safetensors"
        t = load_file(f)
        widths = [int(w) for w in t["widths"]]
        assert a.shape[1] == sum(widths) and b.shape[0] == len(widths), f"layer {i}: expert LoRA does not match the widths"
        delta = down_lora_delta(a, b, widths, args.scale)
        w = t["down"]
        new = (w.float() + delta).to(w.dtype)
        ex_rel.append(float((new.float() - w.float()).norm() / w.float().norm().clamp_min(1e-12)))
        t["down"] = new.contiguous()
        save_file(t, f, metadata={"format": "pt"})
    if layers:
        print(f"expert down_proj LoRA folded into {len(layers)} layers", flush=True)
    rel = rel or [0.0]
    out = {"lora": Path(args.lora).name, "folded": len(mods), "files": len(by_file), "scale": args.scale,
           "expert_layers_folded": len(layers),
           "expert_relative_change_max": max(ex_rel) if ex_rel else 0.0,
           "relative_change_mean": sum(rel) / len(rel), "relative_change_max": max(rel)}
    (d / "fold_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
