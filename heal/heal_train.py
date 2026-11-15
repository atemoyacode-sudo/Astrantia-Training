"""Light healing: train LoRA on the 96x256 student with the teacher's own answers (sequence-level distillation).

Data (what the student learns to write, the teacher's completion after "<think>\\n" up to <|im_end|>):
  R  the teacher's answers to prompts.jsonl (math / science / code), from the teacher kernel (traces.jsonl); only answers
     that close their thinking and stop on their own are used
  G  the teacher's own answers to UltraChat (jiapingW/qwen3.5-35b-a3b-ultrachat-regen, first turn only)
G and R are interleaved so that G is --g-share of the tokens at every point, so every milestone sees the same mix.

Model: the student in float16 split over the GPUs by layer, LoRA (float32) on attention, GDN and shared-expert linear
layers, dynamic loss scaling, gradient checkpointing, the output layer's loss computed --loss-chunk tokens at a time.
At each milestone (tokens fed, prompts included) the LoRA is saved and KL to the teacher on the held-out chat windows
(A, B) is measured. Resumable: the state is saved at milestones and every --save-minutes.

  python heal_train.py --student /kaggle/input/xmodel-heal-student/96x256 --traces traces.jsonl \\
      --ultrachat data-00000-of-00019.arrow --run <dir with data.npz, teacher.npz> --out /kaggle/working/heal
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common import kl_per_seq
from lora_train import ExpertLoRA, LoRALinear

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj",
           "down_proj")


# ---------------------------------------------------------------- data


def clean_completion(text: str) -> str | None:
    """The completion after the generation prompt (which ends with "<think>\\n"), or None if it is unusable."""
    text = text.removeprefix("<think>\n").removeprefix("<think>")
    text = re.sub(r"cw\n</think>", "\n</think>", text)  # an artifact in the UltraChat set ("...done.cw\n</think>")
    if "</think>" not in text or not text.split("</think>", 1)[1].strip():
        return None
    return text


def encode(tok, prompt: str, completion: str, max_len: int, thinking: bool = True):
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=thinking)
    # thinking off: the template closes an empty thinking block itself and the completion is the answer alone
    assert prefix.endswith("<think>\n" if thinking else "</think>\n\n"), prefix[-40:]
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    c = tok(completion + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return {"ids": p + c, "start": len(p)}


EXITS = [
    "I keep writing the same check, and it gives the same thing every time. Repeating it won't change anything, so I'll move on.\n",
    "I'm going in circles here. The step above stands as written; let me continue from it.\n",
    "This is the same verification again. I've already done it, so I'll stop re-checking and keep going.\n",
    "Wait, I'm just repeating myself. Enough checking this part; on to the next step.\n",
]


def loop_exit_example(tok, prompt: str, completion: str, max_len: int, rng: random.Random):
    """A teacher answer with a stretch of its own thinking repeated 3-8 times (the way the cut model loops), followed
    by a sentence that breaks out of the loop and the rest of the answer. Loss only on the break-out and after it."""
    think, rest = completion.split("</think>", 1)
    lines = think.split("\n")
    cands = [i for i in range(int(len(lines) * 0.25), int(len(lines) * 0.85))
             if lines[i - 1].strip() and len(lines[i - 1]) < 300]
    if len(lines) < 8 or not cands:
        return None
    c = rng.choice(cands)
    span = rng.choice([1, 2, 2, 3])
    seg = [ln for ln in lines[max(0, c - span):c] if ln.strip()]
    head = "\n".join(lines[:c]) + "\n"
    loop = ("\n".join(seg) + "\n") * rng.randint(3, 8)
    tail = rng.choice(EXITS) + "\n".join(lines[c:]) + "</think>" + rest + "<|im_end|>\n"
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    p, h, l, t = (tok(x, add_special_tokens=False)["input_ids"] for x in (prefix, head, loop, tail))
    if len(p) + len(h) + len(l) + len(t) > max_len:
        return None
    return {"ids": p + h + l + t, "start": len(p) + len(h) + len(l)}


def load_examples(args, tok):
    rng = random.Random(args.seed)
    r_ex, stats = [], {"R_rows": 0, "R_kept": 0, "G_rows": 0, "G_kept": 0}
    for fi, path in enumerate(args.traces):
        # each file is shuffled on its own and appended in the given order, so a later session that adds files keeps
        # the same sequence prefix (and can resume where the earlier session stopped)
        file_ex = []
        for line in open(path):
            row = json.loads(line)
            stats["R_rows"] += 1
            if row.get("finish_reason") != "stop":
                continue
            if row.get("thinking", True):
                comp = clean_completion(row.get("completion") or "")
                ex = comp and encode(tok, row["prompt"], comp, args.max_len)
            else:  # an answer given with thinking off (README 8)
                comp = (row.get("completion") or "").strip()
                ex = comp and "</think>" not in comp and encode(tok, row["prompt"], comp, args.max_len, thinking=False)
            if ex:
                ex["kind"] = "R/" + row["domain"]
                file_ex.append(ex)
        random.Random(args.seed + 1000 * fi).shuffle(file_ex)
        r_ex += file_ex
    if args.mix_traces:  # one shuffle over all files: a short run then sees every file's answers, not only the first's
        random.Random(args.seed + 99).shuffle(r_ex)
    import pyarrow as pa

    g_ex = []
    with pa.memory_map(args.ultrachat) as src:
        table = pa.ipc.open_stream(src).read_all()
    convs = table.column("conversations").to_pylist()
    rng.shuffle(convs)
    for conv in convs:
        stats["G_rows"] += 1
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        comp = clean_completion(conv[1]["content"] or "")
        ex = comp and encode(tok, conv[0]["content"], comp, args.max_len)
        if ex:
            ex["kind"] = "G"
            g_ex.append(ex)
    stats["R_kept"], stats["G_kept"] = len(r_ex), len(g_ex)
    if not r_ex:  # speed / memory check before the teacher's answers exist
        return g_ex, {**stats, "sequence_tokens": sum(len(e["ids"]) for e in g_ex), "trained_tokens": 0, "by_kind": {}}
    # interleave by tokens so G stays at --g-share everywhere along the sequence
    out, gi, ri, g_tok, all_tok = [], 0, 0, 0, 0
    while ri < len(r_ex) and gi < len(g_ex):
        if g_tok < args.g_share * all_tok:
            ex, gi = g_ex[gi], gi + 1
            g_tok += len(ex["ids"])
        else:
            ex, ri = r_ex[ri], ri + 1
        all_tok += len(ex["ids"])
        out.append(ex)
    if args.loop_exit:  # phase 2, appended after the original sequence (so earlier sessions' positions stay valid):
        # every teacher answer again, now as a loop-exit example, interleaved with UltraChat answers not used yet
        rng2 = random.Random(args.seed + 7)
        rows = [json.loads(line) for path in args.traces for line in open(path)]
        lx = []
        for row in rows:
            comp = clean_completion(row.get("completion") or "") if row.get("finish_reason") == "stop" else None
            ex = comp and loop_exit_example(tok, row["prompt"], comp, args.max_len, rng2)
            if ex:
                ex["kind"] = "X/" + row["domain"]
                lx.append(ex)
        rng2.shuffle(lx)
        stats["loop_exit_examples"], n_phase1 = len(lx), len(out)
        xi, g_tok2, all_tok2 = 0, 0, 0
        while xi < len(lx) and gi < len(g_ex):
            if g_tok2 < args.loop_exit_g_share * all_tok2:
                ex, gi = g_ex[gi], gi + 1
                g_tok2 += len(ex["ids"])
            else:
                ex, xi = lx[xi], xi + 1
            all_tok2 += len(ex["ids"])
            out.append(ex)
        stats["phase1_tokens"], stats["phase1_examples"] = all_tok, n_phase1
        all_tok += all_tok2
    if args.text:  # W: plain text (Wikipedia), loss on every token, mixed in at --text-share of the tokens everywhere
        w_ex = []
        for line in open(args.text):
            row = json.loads(line)
            ids = tok(row["text"], add_special_tokens=False)["input_ids"][: args.text_len]
            if len(ids) >= 64:
                w_ex.append({"ids": ids, "start": 1, "kind": "W/" + row.get("lang", "?")})
        random.Random(args.seed + 11).shuffle(w_ex)
        stats["W_kept"] = len(w_ex)
        mixed, wi, w_tok, all_tok = [], 0, 0, 0
        for ex in out:
            while wi < len(w_ex) and w_tok < args.text_share * all_tok:
                mixed.append(w_ex[wi])
                w_tok += len(w_ex[wi]["ids"])
                all_tok += len(w_ex[wi]["ids"])
                wi += 1
            mixed.append(ex)
            all_tok += len(ex["ids"])
        out = mixed
    stats["sequence_tokens"] = all_tok
    stats["trained_tokens"] = sum(len(e["ids"]) - e["start"] for e in out)
    stats["by_kind"] = {k: sum(len(e["ids"]) for e in out if e["kind"] == k) for k in sorted({e["kind"] for e in out})}
    return out, stats


# ---------------------------------------------------------------- model

ATTN_CHUNK = 1024


def chunked_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """SDPA over query chunks. On a T4, head_dim 256 makes SDPA fall back to the math kernel, which builds the full
    (heads, T, T) score matrix in float32 (5 GB at 9k tokens). Chunks of ATTN_CHUNK queries (each checkpointed when
    training) keep it under 1 GB. Assumes no KV cache (keys = the same positions as the queries)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    q_len = query.shape[2]
    if q_len <= ATTN_CHUNK:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)
    outs = []
    for s in range(0, q_len, ATTN_CHUNK):
        e = min(s + ATTN_CHUNK, q_len)
        if attention_mask is None:
            m = (torch.arange(s, e, device=query.device)[:, None] >= torch.arange(e, device=query.device)[None, :])[None, None]
        else:
            m = attention_mask[:, :, s:e, :e]

        def f(q, k, v, m=m):
            return sdpa_attention_forward(module, q, k, v, m, dropout, scaling, False, **kwargs)[0]

        args = (query[:, :, s:e], key[:, :, :e], value[:, :, :e])
        outs.append(checkpoint(f, *args, use_reentrant=False) if torch.is_grad_enabled() else f(*args))
    return torch.cat(outs, dim=1), None


def gdn_in_float32() -> None:
    """Run the gated delta rule (GDN layers) in float32: in float16 its backward may overflow on long sequences."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m

    orig = m.torch_chunk_gated_delta_rule
    if getattr(orig, "_fp32", False):
        return

    def fp32(query, key, value, g, beta, **kwargs):
        dt = query.dtype
        out, state = orig(query.float(), key.float(), value.float(), g=g.float(), beta=beta.float(), **kwargs)
        return out.to(dt), state

    fp32._fp32 = True
    m.torch_chunk_gated_delta_rule = fp32


def register_chunked_sdpa() -> str:
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    AttentionInterface.register("chunked_sdpa", chunked_sdpa)
    ALL_MASK_ATTENTION_FUNCTIONS.register("chunked_sdpa", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return "chunked_sdpa"


def load_student(path: str, n_gpu: int):
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    cfg = AutoConfig.from_pretrained(path)
    n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
    if n_gpu == 0:  # local test on CPU
        model = Qwen3_5MoeForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
        attach_uneven(model, cfg, path)
        return model.requires_grad_(False)
    per = math.ceil(n_layers / n_gpu)
    dmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": n_gpu - 1, "lm_head": n_gpu - 1}
    dmap.update({f"model.layers.{i}": min(i // per, n_gpu - 1) for i in range(n_layers)})
    model = Qwen3_5MoeForCausalLM.from_pretrained(path, dtype=torch.float16, device_map=dmap,
                                                  attn_implementation=register_chunked_sdpa())
    attach_uneven(model, cfg, path)
    model.requires_grad_(False)
    return model


def attach_uneven(model, cfg, path: str) -> None:
    """An uneven student (README 6h) is an ordinary checkpoint with placeholder experts plus one file per layer."""
    shape = getattr(getattr(cfg, "text_config", cfg), "uneven_shape", None)
    if shape:
        import uneven

        kept = uneven.attach(model, path)
        print(f"uneven experts {shape}: {kept:,} routed neurons kept ({kept / len(model.model.layers):,.0f} per layer)", flush=True)


def add_lora(model, r: int, alpha: float, expert_r: int = 0) -> list[nn.Parameter]:
    """LoRA on the TARGETS linears; with expert_r > 0 also a per-expert LoRA on every routed expert's gate_up / down."""
    n = 0
    for layer in model.model.layers:
        for _, mod in list(layer.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and cname in TARGETS:
                    setattr(mod, cname, LoRALinear(child, r, alpha))
                    n += 1
    if expert_r:
        model.expert_lora = nn.ModuleList(ExpertLoRA(layer.mlp.experts, expert_r, 2 * expert_r) for layer in model.model.layers)
        n += len(model.expert_lora)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA on {n} linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M parameters", flush=True)
    return params


def lora_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def chunk_ce(h, w, labels):
    return F.cross_entropy(F.linear(h, w).float(), labels, reduction="sum")


def seq_loss(model, ids: torch.Tensor, start: int, chunk: int) -> tuple[torch.Tensor, int]:
    """Summed cross-entropy over the completion tokens of one sequence (batch 1)."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    w = model.lm_head.weight
    labels = ids[0, 1:].to(h.device)
    h = h[:-1]
    total, n = h.new_zeros((), dtype=torch.float32), 0
    for s in range(start - 1, h.shape[0], chunk):
        e = min(s + chunk, h.shape[0])
        total = total + checkpoint(chunk_ce, h[s:e], w, labels[s:e], use_reentrant=False)
        n += e - s
    return total, n


@torch.no_grad()
def eval_kl(model, data, teacher, dev0) -> np.ndarray:
    model.eval()
    kls = []
    for k in ("A", "B"):
        for s in range(len(data[k])):
            x = torch.as_tensor(data[k][s : s + 1], device=dev0)
            logits = model(input_ids=x, use_cache=False).logits
            kls.append(float(kl_per_seq(logits, torch.as_tensor(teacher[f"{k}_logp"][s : s + 1], device=logits.device),
                                        torch.as_tensor(teacher[f"{k}_idx"][s : s + 1], device=logits.device))[0]))
    model.train()
    return np.array(kls)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--ultrachat", required=True)
    ap.add_argument("--run", required=True, help="has data.npz and teacher.npz (held-out chat windows A, B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--milestones", default="1e6,3e6,1e7", help="tokens fed (prompts included)")
    ap.add_argument("--g-share", type=float, default=0.3)
    ap.add_argument("--loop-exit", action="store_true", help="append a phase of loop-exit examples (see loop_exit_example)")
    ap.add_argument("--loop-exit-g-share", type=float, default=0.3)
    ap.add_argument("--jump-to-loop-exit", action="store_true", help="on resume, skip the rest of phase 1")
    ap.add_argument("--mix-traces", action="store_true", help="shuffle the answers of all trace files together "
                                                              "(a new sequence: not for resuming an older session)")
    ap.add_argument("--init-lora", default=None, help="start from these LoRA weights (lora_*.pt or state.pt) with a new "
                                                      "optimizer and the data from its beginning, e.g. on another student")
    ap.add_argument("--max-start-kl", type=float, default=float("inf"), help="with --init-lora: stop before training if the "
                                                                              "student's KL without the LoRA is above this")
    ap.add_argument("--text", default=None, help="jsonl of {text, lang}: plain text trained on every token (W)")
    ap.add_argument("--text-share", type=float, default=0.25, help="share of the tokens that are W, at every point")
    ap.add_argument("--text-len", type=int, default=2048, help="W pieces are cut to this many tokens")
    ap.add_argument("--wiki-run", default=None, help="dir with data.npz + teacher.npz of Wikipedia windows (A, B): "
                    "KL there is measured at each milestone too")
    ap.add_argument("--max-len", type=int, default=9216)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--expert-r", type=int, default=0, help="also LoRA on the routed experts (rank); 0 = off")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--step-tokens", type=int, default=32768, help="trained tokens per optimizer step")
    ap.add_argument("--loss-chunk", type=int, default=1024)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--save-minutes", type=float, default=30.0)
    ap.add_argument("--resume", default=None, help="state.pt from an earlier session")
    ap.add_argument("--smoke-steps", type=int, default=0, help="stop after N optimizer steps (speed / memory check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gdn-fp16", action="store_true", help="keep the GDN delta rule in float16 (default: float32)")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    milestones = [int(float(m)) for m in args.milestones.split(",")]
    log_f = open(out / "train.log", "a")

    def log(m):
        s = time.strftime("%H:%M:%S ") + m
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    examples, dstats = load_examples(args, tok)
    log(f"data: {json.dumps(dstats)}")
    if dstats["sequence_tokens"] < milestones[-1]:
        log(f"only {dstats['sequence_tokens']:,} tokens of data; the last milestone {milestones[-1]:,} will not be reached")

    n_gpu = torch.cuda.device_count()
    if not args.gdn_fp16:
        gdn_in_float32()
    model = load_student(args.student, n_gpu)
    params = add_lora(model, args.r, 2 * args.r, args.expert_r)
    dev0 = "cuda:0" if n_gpu else "cpu"
    run = Path(args.run)
    data, teacher = dict(np.load(run / "data.npz")), dict(np.load(run / "teacher.npz"))
    wiki = None
    if args.wiki_run:
        wiki = (dict(np.load(Path(args.wiki_run) / "data.npz")), dict(np.load(Path(args.wiki_run) / "teacher.npz")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    # Loss scaling per sequence: each sequence's gradient is checked on its own, and a sequence whose float16 backward
    # overflows is dropped (and the scale halved) instead of throwing away the whole step (GradScaler's behaviour).
    accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]

    st = {"pos": 0, "step": 0, "fed": 0, "trained": 0, "next_ms": 0, "results": [], "losses": [], "seconds": 0.0,
          "scale": 1024.0, "good_streak": 0, "skipped": []}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        for k, v in saved["lora"].items():
            named[k].data.copy_(v.to(named[k].device))
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        st = {"scale": float(saved.get("scaler", {}).get("scale", 1024.0)), "good_streak": 0, "skipped": [], **saved["st"]}
        log(f"resumed at step {st['step']}, {st['fed']:,} tokens fed, loss scale {st['scale']:.0f}")
        if args.jump_to_loop_exit and st["pos"] < dstats.get("phase1_examples", 0):
            log(f"jumping from example {st['pos']} to the loop-exit phase at {dstats['phase1_examples']}")
            st["jumped_from"], st["pos"] = st["pos"], dstats["phase1_examples"]
    else:
        kl_raw = None
        if args.init_lora:
            t = time.time()
            kl_raw = eval_kl(model, data, teacher, dev0)  # the LoRA is still zero here: the student as it was cut
            log(f"student without a LoRA: KL A+B {kl_raw.mean():.4f} ({time.time() - t:.0f}s)")
            if kl_raw.mean() > args.max_start_kl:
                raise SystemExit(f"KL {kl_raw.mean():.4f} is above --max-start-kl {args.max_start_kl}: the student looks broken; stopping")
            saved = torch.load(args.init_lora, map_location="cpu", weights_only=False)
            saved = saved.get("lora", saved)
            named = {k: v for k, v in model.named_parameters() if v.requires_grad}
            assert set(saved) == set(named), f"LoRA keys differ: {sorted(set(saved) ^ set(named))[:6]}"
            for k, v in saved.items():
                named[k].data.copy_(v.to(named[k].device))
            log(f"started from the LoRA in {args.init_lora} ({len(saved)} tensors)")
        t = time.time()
        kl0 = eval_kl(model, data, teacher, dev0)
        st["results"].append({"tokens_fed": 0, "tokens_trained": 0, "kl_AB": float(kl0.mean()), "kl_per_seq": kl0.tolist()})
        if wiki:
            st["results"][0]["wiki_kl_AB"] = float(eval_kl(model, *wiki, dev0).mean())
            log(f"step 0: Wikipedia KL A+B {st['results'][0]['wiki_kl_AB']:.4f}")
        if kl_raw is not None:
            st["results"][0].update({"kl_AB_no_lora": float(kl_raw.mean()), "kl_per_seq_no_lora": kl_raw.tolist()})
        log(f"step 0: KL A+B {kl0.mean():.4f} ({time.time() - t:.0f}s)")
        (out / "results.json").write_text(json.dumps(st["results"], indent=1))

    def save_state(tag="state"):
        torch.save({"lora": lora_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st,
                    "args": vars(args)}, out / f"{tag}.pt")

    def backward_one(ids, start) -> tuple[float, int, bool]:
        """Backward one sequence at the current scale; add its unscaled gradient to accum if it is finite."""
        loss_sum, n = seq_loss(model, ids, start, args.loss_chunk)
        (loss_sum * (st["scale"] / args.step_tokens)).backward()
        grads = [p.grad for p in params]
        ok = bool(torch.isfinite(loss_sum).item()) and all(bool(torch.isfinite(g).all()) for g in grads if g is not None)
        if ok:
            for a, g in zip(accum, grads):
                if g is not None:
                    a.add_(g.to(a.device), alpha=1.0 / st["scale"])
        for p in params:
            p.grad = None
        return float(loss_sum.detach()), n, ok

    def optimizer_step(acc_tok) -> float:
        for p, a in zip(params, accum):  # summed over acc_tok tokens but divided by step_tokens
            p.grad = a * (args.step_tokens / acc_tok)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        for p, a in zip(params, accum):
            p.grad = None
            a.zero_()
        st["step"] += 1
        return gnorm

    if args.smoke_steps:  # the longest sequence allowed, to see whether it fits
        long_ids = [t for e in examples for t in e["ids"]][: args.max_len]
        torch.cuda.reset_peak_memory_stats() if n_gpu else None
        t = time.time()
        _, _, ok = backward_one(torch.tensor([long_ids], device=dev0), 1)
        for a in accum:
            a.zero_()
        log(f"one {len(long_ids)}-token sequence: forward+backward {time.time() - t:.1f}s (finite {ok}), peak memory "
            + " ".join(f"{torch.cuda.max_memory_allocated(i) / 2**30:.1f}" for i in range(n_gpu)) + " GB")
    model.train()
    t_sess, t_save, acc_tok, acc_loss, t_log, tok_log, gnorm = time.time(), time.time(), 0, 0.0, time.time(), 0, 0.0
    skipped_at_log = len(st["skipped"])
    done = timed_out = False
    while st["pos"] < len(examples) and not done:
        ex = examples[st["pos"]]
        st["pos"] += 1
        try:
            loss, n, ok = backward_one(torch.tensor([ex["ids"]], device=dev0), ex["start"])
        except torch.OutOfMemoryError:  # a bigger student leaves less room: drop this sequence, keep the session
            for p in params:
                p.grad = None
            torch.cuda.empty_cache()
            loss, n, ok = 0.0, 0, None
        st["fed"] += len(ex["ids"])
        tok_log += len(ex["ids"])
        if ok is None:
            st["skipped"].append({"pos": st["pos"] - 1, "kind": ex["kind"], "len": len(ex["ids"]), "oom": True})
            log(f"out of memory on a {len(ex['ids'])}-token sequence; skipped")
        elif ok:
            acc_tok += n
            acc_loss += loss
            st["trained"] += n
            st["good_streak"] += 1
            if st["good_streak"] % 100 == 0:
                st["scale"] = min(st["scale"] * 2, 2.0 ** 15)
        else:
            st["skipped"].append({"pos": st["pos"] - 1, "kind": ex["kind"], "len": len(ex["ids"]), "scale": st["scale"]})
            st["scale"] = max(st["scale"] / 2, 1.0)
            st["good_streak"] = 0
        if acc_tok >= args.step_tokens:
            gnorm = optimizer_step(acc_tok)
            st["losses"].append(acc_loss / acc_tok)
            if st["step"] % 5 == 0:
                el = time.time() - t_log
                log(f"step {st['step']}: loss {np.mean(st['losses'][-5:]):.4f} gnorm {gnorm:.2f} scale {st['scale']:.0f} "
                    f"skipped {len(st['skipped']) - skipped_at_log} seqs fed {st['fed']:,} ({tok_log / el:.0f} tok/s) mem "
                    + " ".join(f"{torch.cuda.max_memory_allocated(i) / 2**30:.1f}" for i in range(n_gpu)) + " GB")
                t_log, tok_log, skipped_at_log = time.time(), 0, len(st["skipped"])
            acc_tok, acc_loss = 0, 0.0
            if args.smoke_steps and st["step"] >= args.smoke_steps:
                done = True
        reached = st["next_ms"] < len(milestones) and st["fed"] >= milestones[st["next_ms"]] and acc_tok == 0
        if reached:
            m = milestones[st["next_ms"]]
            st["next_ms"] += 1
            kl = eval_kl(model, data, teacher, dev0)
            st["results"].append({"tokens_fed": st["fed"], "tokens_trained": st["trained"], "milestone": m,
                                  "step": st["step"], "kl_AB": float(kl.mean()), "kl_per_seq": kl.tolist(),
                                  "skipped_seqs": len(st["skipped"])})
            if wiki:
                st["results"][-1]["wiki_kl_AB"] = float(eval_kl(model, *wiki, dev0).mean())
                log(f"milestone {m:,}: Wikipedia KL A+B {st['results'][-1]['wiki_kl_AB']:.4f}")
            torch.save(lora_state(model), out / f"lora_{m:.0e}.pt".replace("+0", ""))
            (out / "results.json").write_text(json.dumps(st["results"], indent=1))
            log(f"milestone {m:,}: KL A+B {kl.mean():.4f} (start {st['results'][0]['kl_AB']:.4f}), saved LoRA; "
                f"{len(st['skipped'])} sequences skipped so far")
            save_state()
            t_save = time.time()
            if st["next_ms"] == len(milestones):
                done = True
        if time.time() - t_save > args.save_minutes * 60 and acc_tok == 0:
            save_state()
            t_save = time.time()
        if time.time() - t_start > args.max_hours * 3600 and acc_tok == 0:
            log("time limit reached")
            done = timed_out = True
    st["seconds"] += time.time() - t_sess
    if st["pos"] >= len(examples) and acc_tok > 0:  # apply what is left of the last, partial step
        optimizer_step(acc_tok)
        acc_tok = 0
    if st["pos"] >= len(examples) and st["step"] > 0:  # the data ran out before the last milestone
        kl = eval_kl(model, data, teacher, dev0)
        st["results"].append({"tokens_fed": st["fed"], "tokens_trained": st["trained"], "milestone": "end_of_data",
                              "step": st["step"], "kl_AB": float(kl.mean()), "kl_per_seq": kl.tolist()})
        torch.save(lora_state(model), out / "lora_final.pt")
        (out / "results.json").write_text(json.dumps(st["results"], indent=1))
        log(f"end of data at {st['fed']:,} tokens: KL A+B {kl.mean():.4f}, saved lora_final.pt")
    elif timed_out and st["step"] > 0:  # a session cut by --max-hours still leaves a measured, usable LoRA
        kl = eval_kl(model, data, teacher, dev0)
        st["results"].append({"tokens_fed": st["fed"], "tokens_trained": st["trained"], "milestone": "time_limit",
                              "step": st["step"], "kl_AB": float(kl.mean()), "kl_per_seq": kl.tolist(),
                              "skipped_seqs": len(st["skipped"])})
        torch.save(lora_state(model), out / "lora_final.pt")
        (out / "results.json").write_text(json.dumps(st["results"], indent=1))
        log(f"time limit at {st['fed']:,} tokens: KL A+B {kl.mean():.4f} (start {st['results'][0]['kl_AB']:.4f}), saved lora_final.pt")
    save_state()
    log(f"stopped at step {st['step']}, {st['fed']:,} tokens fed, {st['trained']:,} trained; results in {out}/results.json")


if __name__ == "__main__":
    main()
