"""An uneven student (uneven.py) written as an ordinary Qwen3.5-MoE checkpoint that vLLM and plain transformers load.

Every layer gets the same number of experts and one width: an expert's missing neurons are zero rows (a zero gate row
gives silu(0) * up = 0, so they add nothing), and a layer with fewer experts than the largest layer gets dummy experts
with zero weights. A dummy must never be picked, and the router has no bias, so its router row is -c * u, where u is
a direction with a positive cosine to that layer's router input on every calibration token (the one with the largest
smallest cosine): the logit is -c * (u . h). c is the smallest value that puts the dummy's logit `--margin` under
the 8th-largest real logit on every calibration token. This is checked,
not proven: the script counts dummy picks and compares the logits with the uneven model on held-out windows.

A LoRA from heal_train.py (lora_*.pt) is folded into the weights first.

  python pad_uneven.py --student /content/student/g128b28160 --lora lora_final.pt --run <dir with data.npz, teacher.npz> \\
      --out /content/models/new [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "heal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def fold_lora(model, lora_path: str, r: int) -> int:
    from heal_train import add_lora
    from lora_train import LoRALinear

    add_lora(model, r, 2 * r)
    named = dict(model.named_parameters())
    state = torch.load(lora_path, map_location="cpu", weights_only=False)
    state = state.get("lora", state)
    assert set(state) == {k for k, v in named.items() if v.requires_grad}, "the LoRA does not match the model"
    for k, v in state.items():
        named[k].data.copy_(v.to(named[k].device))
    n = 0
    for layer in model.model.layers:
        for _, mod in list(layer.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, LoRALinear):
                    w = child.base.weight
                    delta = (child.B.float() @ child.A.float()) * child.scale
                    w.data = (w.data.float() + delta.to(w.device)).to(w.dtype)
                    setattr(mod, cname, child.base)
                    n += 1
    model.requires_grad_(False)
    return n


@torch.no_grad()
def router_stats(model, batches: list, dev: str, top_k: int, need: set[int]):
    """Per layer in `need`: the router inputs of every calibration token (N, H) and the top_k-th largest real logit (N,)."""
    hs, l8 = {i: [] for i in need}, {i: [] for i in need}

    def hook(i):
        def fn(mod, args):
            h = args[0].reshape(-1, args[0].shape[-1])
            hs[i].append(h.to(torch.float16).cpu())  # kept on the CPU: on 2x T4 there is no room for them
            logits = h.float() @ mod.weight.float().T
            l8[i].append(logits.topk(min(top_k, logits.shape[1]), dim=-1).values[:, -1].cpu())
        return fn

    handles = [l.mlp.gate.register_forward_pre_hook(hook(i)) for i, l in enumerate(model.model.layers) if i in need]
    for x in batches:
        model(input_ids=torch.as_tensor(x, device=next(model.parameters()).device).reshape(1, -1), use_cache=False)
    for h in handles:
        h.remove()
    return {i: torch.cat(v) for i, v in hs.items()}, {i: torch.cat(v) for i, v in l8.items()}


@torch.no_grad()
def safe_direction(h: torch.Tensor, iters: int = 2000) -> tuple[torch.Tensor, float]:
    """A unit vector u with the largest smallest cosine to every row of h (the min-norm point of the rows' convex
    hull, Gilbert's algorithm), and that smallest cosine. It is positive only if all rows lie in one half-space."""
    hn = torch.nn.functional.normalize(h.float(), dim=-1)
    w = hn.mean(0)
    for _ in range(iters):
        proj = hn @ w
        t = int(proj.argmin())
        ww = float(w @ w)
        if float(proj[t]) >= ww * (1 - 1e-3):
            break
        z = hn[t]
        a = min(1.0, max(0.0, (ww - float(proj[t])) / float((w - z) @ (w - z))))
        w = (1 - a) * w + a * z
    u = w / w.norm()
    return u, float((hn @ u).min())


def prompt_ids(tok, path: str, limit: int) -> list:
    """Chat-formatted prompts from an eval_chat.py task file, thinking on and off in turn: these start at position 0,
    which the 512-token windows do not."""
    out = []
    for n, line in enumerate(open(path)):
        if n >= limit:
            break
        text = tok.apply_chat_template([{"role": "user", "content": json.loads(line)["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=n % 2 == 0)
        out.append(np.array(tok(text, add_special_tokens=False)["input_ids"][:2048]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--run", required=True, help="data.npz (C = calibration windows, A, B = held-out) and teacher.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dtype", default="bfloat16")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--cal-prompts", default="", help="an eval_chat.py task file: its prompts join the calibration tokens")
    ap.add_argument("--check-prompts", default="", help="another task file: its prompts join the held-out check")
    ap.add_argument("--max-cal-prompts", type=int, default=300)
    ap.add_argument("--no-check", action="store_true", help="skip the held-out check (it loads the padded model, too big for 2x T4)")
    ap.add_argument("--allow-unsafe", action="store_true", help="write the checkpoint even if no safe direction exists (tests)")
    ap.add_argument("--width-multiple", type=int, default=1, help="round the common width up (the tiny CPU test needs 4)")
    args = ap.parse_args()
    from heal_train import eval_kl, load_student, register_chunked_sdpa
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    student, out, dev = Path(args.student), Path(args.out), args.device
    sdt = getattr(torch, args.save_dtype)
    data = np.load(Path(args.run) / "data.npz")
    model = load_student(str(student), torch.cuda.device_count() if dev.startswith("cuda") else 0).eval()
    if args.lora:
        print(f"folded {fold_lora(model, args.lora, args.r)} LoRA layers", flush=True)
    conf = json.loads((student / "config.json").read_text())
    top_k = conf["num_experts_per_tok"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(student)
    cal = list(data["C"][: args.max_windows] if args.max_windows else data["C"])
    cal += prompt_ids(tok, args.cal_prompts, args.max_cal_prompts) if args.cal_prompts else []
    packs = [load_file(student / f"uneven_{i:03d}.safetensors") for i in range(len(model.model.layers))]
    e_max = max(p["gate"].shape[0] for p in packs)
    need = {i for i, p in enumerate(packs) if p["gate"].shape[0] < e_max}
    hs, l8 = router_stats(model, cal, dev, top_k, need)
    us, cs, cos_min = {}, {}, {}
    for i in sorted(need):
        us[i], cos_min[i] = safe_direction(hs[i])
        assert cos_min[i] > 0 or args.allow_unsafe, f"layer {i}: no direction is positive on every calibration token ({cos_min[i]:.4f})"
        s = hs[i].float() @ us[i]
        # the dummy's logit -c * (u . h) lies `margin` under the top_k-th real logit on every calibration token
        cs[i] = float(((args.margin - l8[i]) / s.clamp_min(1e-6)).clamp_min(0).max()) if cos_min[i] > 0 else 1.0
    del hs, l8
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    w_max = max(int(p["widths"].max()) for p in packs)
    w_max = -(-w_max // args.width_multiple) * args.width_multiple
    hidden = packs[0]["gate"].shape[1]
    out.mkdir(parents=True, exist_ok=True)
    weight_map, dummies, c_used = {}, 0, []
    for i, (layer, p) in enumerate(zip(model.model.layers, packs)):
        e = p["gate"].shape[0]
        gate = torch.zeros(e_max, hidden)
        gate[:e] = p["gate"].float()
        if e < e_max:
            gate[e:] = -cs[i] * us[i].cpu()
            dummies += e_max - e
            c_used.append(cs[i])
        gate_up = torch.zeros(e_max, 2 * w_max, hidden)
        down = torch.zeros(e_max, hidden, w_max)
        o = 0
        for j, w in enumerate(p["widths"].tolist()):
            gate_up[j, :w] = p["gate_up"][2 * o : 2 * o + w].float()
            gate_up[j, w_max : w_max + w] = p["gate_up"][2 * o + w : 2 * o + 2 * w].float()
            down[j, :, :w] = p["down"][:, o : o + w].float()
            o += w
        sd = {k: v for k, v in layer.state_dict().items() if not k.startswith(("mlp.experts.", "mlp.gate."))}
        sd.update({"mlp.gate.weight": gate, "mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down})
        fname = f"layer_{i:03d}.safetensors"
        save_file({f"model.layers.{i}.{k}": v.to(sdt).contiguous().cpu() for k, v in sd.items()}, out / fname)
        weight_map.update({f"model.layers.{i}.{k}": fname for k in sd})
        del gate_up, down, sd
    shutil.copy(student / "rest.safetensors", out / "rest.safetensors")
    old_map = json.loads((student / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map.update({k: v for k, v in old_map.items() if v == "rest.safetensors"})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    shape = conf.get("uneven_shape")
    conf = {k: v for k, v in conf.items() if not k.startswith("uneven_")}
    conf.update({"num_experts": e_max, "moe_intermediate_size": w_max, "torch_dtype": args.save_dtype})
    (out / "config.json").write_text(json.dumps(conf, indent=1))
    for f in student.iterdir():
        if f.suffix in (".jinja", ".txt") or f.name.startswith("tokenizer") or f.name in ("vocab.json", "generation_config.json", "merges.txt"):
            shutil.copy(f, out / f.name)
    print(f"{shape}: wrote {out}: {e_max} experts of width {w_max} in every layer, {dummies} dummy experts in "
          f"{len(c_used)} layers (c from {min(c_used, default=0):.2f} to {max(c_used, default=0):.2f}; smallest cosine "
          f"of a calibration token to its layer's direction {min(cos_min.values(), default=1):.4f}; {len(cal)} calibration "
          f"sequences)", flush=True)

    # ---- the cheap check, on the uneven model itself: on held-out sequences, would a dummy's logit reach the top_k?
    held = list(np.concatenate([data[k] for k in ("A", "B")])) if "A" in data.files else cal
    if args.max_windows:
        held = held[: args.max_windows]
    held += prompt_ids(tok, args.check_prompts, 300) if args.check_prompts else []
    hit, tokens = [0], [0]

    def watch(i):
        def fn(mod, a):
            h = a[0].reshape(-1, a[0].shape[-1]).float()
            kth = (h @ mod.weight.float().T).topk(top_k, dim=-1).values[:, -1]
            dummy = -cs[i] * (h @ us[i].to(h.device).float())
            hit[0] += int((dummy >= kth).sum())
            tokens[0] += h.shape[0] if i == min(need) else 0
        return fn

    hooks = [l.mlp.gate.register_forward_pre_hook(watch(i)) for i, l in enumerate(model.model.layers) if i in need]
    with torch.no_grad():
        for seq in held:
            model(input_ids=torch.as_tensor(seq, device=next(model.parameters()).device).reshape(1, -1), use_cache=False)
    for h in hooks:
        h.remove()
    print(f"dummy check on {len(held)} held-out sequences ({tokens[0]:,} tokens, {len(need)} layers): a dummy would "
          f"reach the top {top_k} {hit[0]} times", flush=True)
    if args.no_check:
        (out / "pad_check.json").write_text(json.dumps({"experts": e_max, "width": w_max, "dummies": dummies, "dummy_hits": hit[0],
                                                        "cos_min": min(cos_min.values(), default=1), "c": cs}, indent=1))
        return
    # ---- check on held-out windows: no dummy is picked, and the logits are the uneven model's
    kw = {"dtype": torch.float16, "device_map": dev, "attn_implementation": register_chunked_sdpa()} if dev.startswith("cuda") else \
         {"dtype": torch.float32, "attn_implementation": "sdpa"}
    padded = Qwen3_5MoeForCausalLM.from_pretrained(out, **kw).eval().requires_grad_(False)
    real = [p["gate"].shape[0] for p in packs]
    picked = [0]

    def count(i):
        def fn(mod, args, output):
            picked[0] += int((output[2] >= real[i]).sum())
        return fn

    for i, l in enumerate(padded.model.layers):
        l.mlp.gate.register_forward_hook(count(i))
    worst, agree, n = 0.0, 0, 0
    with torch.no_grad():
        for seq in held:
            x = torch.as_tensor(seq, device=dev).reshape(1, -1)
            a, b = model(input_ids=x, use_cache=False).logits.float(), padded(input_ids=x, use_cache=False).logits.float()
            worst = max(worst, float((a - b).abs().max()))
            agree += int((a.argmax(-1) == b.argmax(-1)).sum())
            n += a.shape[1]
    print(f"check on {len(held)} held-out sequences: dummy experts picked {picked[0]} times; max |logit difference| "
          f"{worst:.3e}; same top token on {agree}/{n} positions", flush=True)
    tpath = Path(args.run) / "teacher.npz"
    if tpath.exists() and "A" in data.files and not args.max_windows:
        teacher = np.load(tpath)
        print(f"KL A+B to the original: uneven {eval_kl(model, data, teacher, dev).mean():.4f}, padded "
              f"{eval_kl(padded, data, teacher, dev).mean():.4f}", flush=True)
    (out / "pad_check.json").write_text(json.dumps({"dummy_picks": picked[0], "max_logit_diff": worst, "same_top": [agree, n],
                                                    "experts": e_max, "width": w_max, "dummies": dummies,
                                                    "cos_min": min(cos_min.values(), default=1), "c": cs}, indent=1))


if __name__ == "__main__":
    main()
