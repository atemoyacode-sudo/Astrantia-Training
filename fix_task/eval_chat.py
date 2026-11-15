"""Does a repaired model work as a chat model again? Ask it the 60 fixed questions (20 science / math / code) with
thinking on, then grade it and show it next to the reference models on the same questions.

Besides accuracy, it reports the ways the cut models break:
  think_closed  the answer reaches </think> (the cut models almost never do)
  boxed         the final part has a \\boxed{...}
  truncated     hit --max-tokens before finishing
  echo          the response starts by repeating the question
  looping       the end of the response is a loop (it compresses below 15% with zlib)
  think_loop    "<think>" appears 3+ times (the cut model repeats it forever)

Backends (pick one):
  --base-url http://127.0.0.1:8000/v1 --model-name x   an OpenAI-compatible server (vLLM: fast; recommended)
  --model /path/to/checkpoint                            Hugging Face transformers (slow; use --quick)
  --responses answers.jsonl                              only grade answers you already have

  # vLLM (G4/A100):  vllm serve /path/ckpt --served-model-name fixed --max-model-len 20480 --max-num-seqs 64
  python eval_chat.py --base-url http://127.0.0.1:8000/v1 --model-name fixed --out fixed.jsonl --allow-exec
  # quick format check without a server (3 questions per domain, 4k tokens; minutes on a GPU)
  python eval_chat.py --model /path/ckpt --quick --out quick.jsonl
  # Mac (mlx-lm, e.g. 4-bit):  mlx_lm.server --model ~/models/x_4bit --port 8080
  python eval_chat.py --base-url http://127.0.0.1:8080/v1 --model-name x --no-seed --concurrency 8 --out x.jsonl

Code answers are graded by running them, so --allow-exec is needed and it should run on a disposable VM.
The reference answers (orig = Qwen3.5-35B-A3B, q9b, q4b, pruned48 = cut to 48 experts, lin_exp = cut + LoRA) were made
with the same settings: thinking on, temperature 0.6, top-p 0.95, top-k 20, seed 0 per request, up to 16,000 tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
import grade_bench  # noqa: E402

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


# ---------------------------------------------------------------- asking the model


def ask_server(args, tasks):
    import requests

    def one(t):
        body = {"model": args.model_name, "messages": [{"role": "user", "content": t["prompt"]}],
                "max_tokens": args.max_tokens, "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
                "top_k": SAMPLING["top_k"], "chat_template_kwargs": {"enable_thinking": not args.no_thinking}}
        if args.temperature is not None:  # 0 = greedy: the model's single most likely answer
            body["temperature"] = args.temperature
        if args.top_p is not None:
            body["top_p"] = args.top_p
        if not args.no_seed:
            body["seed"] = 0
        if args.rep_penalty:
            body["repetition_penalty"], body["repetition_context_size"] = args.rep_penalty, args.rep_context
        if args.presence_penalty:
            body["presence_penalty"] = args.presence_penalty
        for attempt in range(4):
            try:
                r = requests.post(f"{args.base_url}/chat/completions", json=body, timeout=args.timeout)
                r.raise_for_status()
                j = r.json()
                ch, msg = j["choices"][0], j["choices"][0]["message"]
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")  # vLLM / mlx-lm
                if reasoning:  # a server that splits off the thinking: put it back, closed only if it was closed
                    closed = bool(text) or ch.get("finish_reason") != "length"
                    text = f"<think>{reasoning}{'</think>' if closed else ''}{text}"
                return {"id": t["id"], "task": t["task"], "text": text, "finish_reason": ch.get("finish_reason"),
                        "completion_tokens": j.get("usage", {}).get("completion_tokens"), "thinking": True}
            except Exception as e:  # busy server or a dropped connection
                err = repr(e)
        return {"id": t["id"], "task": t["task"], "text": "", "error": err, "thinking": True}

    # each answer is appended to <out>.partial as it arrives, so a crash loses nothing; --resume skips answered ones
    import threading
    lock, part = threading.Lock(), Path(str(args.out) + ".partial")

    def one_saved(t):
        r = one(t)
        if not r.get("error"):
            with lock, open(part, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    with ThreadPoolExecutor(args.concurrency) as pool:
        return list(pool.map(one_saved, tasks))


def ask_transformers(args, tasks):
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    cfg = transformers.AutoConfig.from_pretrained(args.model)
    cls = getattr(transformers, (cfg.architectures or [""])[0], None) or transformers.AutoModelForCausalLM
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = cls.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    out = []
    for t in tasks:
        prompt = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(0)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=True, **SAMPLING,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=False)
        hit_eos = tok.eos_token_id in new.tolist() or "<|im_end|>" in text
        text = text.split("<|im_end|>")[0]
        if prompt.rstrip().endswith("<think>"):
            text = "<think>\n" + text  # the template opened the thinking block itself
        out.append({"id": t["id"], "task": t["task"], "text": text, "finish_reason": "stop" if hit_eos else "length",
                    "completion_tokens": int(len(new)), "thinking": True})
        print(f"  {t['id']}: {len(new)} tokens", flush=True)
    return out


# ---------------------------------------------------------------- grading


def lenient_sci(text, answer):
    t = text.split("</think>")[-1]
    m = (re.findall(r"\\boxed\{\s*([A-J])", t) or re.findall(r"(?:answer is|choose|Answer:)\s*\**\(?([A-J])\b", t)
         or re.findall(r"^\s*\**([A-J])[\.\)]", t, re.M))
    return bool(m) and m[-1] == answer


def lenient_math(text, answer):
    t = text.split("</think>")[-1]
    b = grade_bench.last_boxed(t, raw=True)
    if b is None:
        return False
    return grade_bench.grade_math({"answer": answer}, {"text": "</think>" + b.join(["\\boxed{", "}"]), "thinking": True})


def lenient_code(task, text, allow_exec):
    if not allow_exec:
        return None
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks:
        return False
    return grade_bench.run_code(blocks[-1] + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n", 30.0)


def loose(task, text, allow_exec):
    """The last answer found anywhere in the response, thinking included (for answers that loop past the token limit):
    sci: last \\boxed{X} / "answer is X" / "Answer: X"; math: last \\boxed{...}; code: last ```python block that
    defines the entry point (else the last block), run against the tests."""
    kind = task["task"]
    if kind == "sci":
        hits = [(m.start(), m.group(1)) for pat in (r"\\boxed\{\s*\(?([A-J])\b", r"(?:answer is|Answer:|answer:)\s*\**\(?([A-J])\b")
                for m in re.finditer(pat, text)]
        return bool(hits) and max(hits)[1] == task["answer"]
    if kind == "math":
        b = grade_bench.last_boxed(text, raw=True)
        return b is not None and grade_bench.grade_math({"answer": task["answer"]},
                                                        {"text": "</think>\\boxed{" + b + "}", "thinking": True})
    if not allow_exec:
        return None
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks:
        return False
    named = [b for b in blocks if f"def {task['entry_point']}" in b]
    prog = (named or blocks)[-1]
    return grade_bench.run_code(prog + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n", 30.0)


def format_flags(task, row):
    text = row.get("text") or ""
    body = re.sub(r"\s+", " ", text.replace("<think>", "").lstrip())[:150]
    q = re.sub(r"\s+", " ", task["prompt"])[:300]
    # echo: an 80-character stretch of the question within the first 150 characters of the response
    # (0/180 for the healthy references, 42/120 for the cut ones)
    echo = any(q[i : i + 80] in body for i in range(0, max(1, len(q) - 80), 10)) if len(q) >= 80 else False
    # looping: the last 4,000 characters compress below 15% with zlib (a loop of any period compresses very well).
    # 2-5 of 60 for the healthy references (their rare runaway answers), 53-54 of 60 for the cut ones
    tail = text[-4000:].encode()
    repeating = len(tail) > 200 and len(zlib.compress(tail, 9)) / len(tail) < 0.15
    final = text.split("</think>")[-1] if "</think>" in text else ""
    return {"think_closed": "</think>" in text, "boxed": "\\boxed" in final,
            "truncated": row.get("finish_reason") == "length", "echo": echo, "looping": repeating,
            "think_loop": text.count("<think>") >= 3}


def grade_rows(rows, tasks, allow_exec):
    out = {}
    for r in rows:
        t = tasks[r["id"]]
        if t["task"] == "code":
            strict = grade_bench.grade_code(t, r) if allow_exec else None
            lenient = lenient_code(t, r.get("text") or "", allow_exec)
        else:
            strict = grade_bench.GRADERS[t["task"]](t, r)
            lenient = (lenient_sci if t["task"] == "sci" else lenient_math)(r.get("text") or "", t["answer"])
        out[r["id"]] = {"task": t["task"], "strict": strict, "lenient": lenient,
                        "loose": loose(t, r.get("text") or "", allow_exec), "tokens": r.get("completion_tokens") or 0,
                        **format_flags(t, r)}
    return out


def table(graded_by_model):
    cols = ["strict", "lenient", "loose", "think_closed", "boxed", "truncated", "echo", "looping", "think_loop"]
    print(f"\n{'model':10s} {'dom':5s} " + " ".join(f"{c:>12s}" for c in cols) + f" {'tokens':>7s}")
    for name, g in graded_by_model.items():
        for dom in ("sci", "math", "code"):
            vals = [v for v in g.values() if v["task"] == dom]
            if not vals:
                continue
            cells = []
            for c in cols:
                xs = [v[c] for v in vals if v[c] is not None]
                cells.append(f"{sum(xs):>5d}/{len(xs):<6d}" if xs else f"{'-':>12s}")
            print(f"{name:10s} {dom:5s} " + " ".join(cells) + f" {sum(v['tokens'] for v in vals) / len(vals):7.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model-name", default="fixed")
    ap.add_argument("--model", help="local checkpoint or HF repo, for the transformers backend")
    ap.add_argument("--tokenizer", help="defaults to --model (e.g. Qwen/Qwen3.5-35B-A3B if the checkpoint lacks one)")
    ap.add_argument("--responses", help="grade an existing answers file instead of asking a model")
    ap.add_argument("--out", default="answers.jsonl")
    ap.add_argument("--n-per-domain", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="3 questions per domain, up to 4,096 tokens")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=3600, help="seconds to wait for one answer (slow GPUs need more)")
    ap.add_argument("--resume", action="store_true", help="keep answers already in --out / --out.partial, ask only the rest")
    ap.add_argument("--no-seed", action="store_true", help="send no seed (mlx-lm's server batches only unseeded requests)")
    ap.add_argument("--device")
    ap.add_argument("--allow-exec", action="store_true", help="run the code answers (only on a disposable VM)")
    ap.add_argument("--refs", default=str(HERE / "reference_answers.jsonl"))
    ap.add_argument("--tasks", default=str(HERE / "chat_eval_set.jsonl"),
                    help="question set; check_set.jsonl = 60 questions nothing has been tuned on")
    ap.add_argument("--rep-penalty", type=float, default=0.0, help="repetition penalty (mlx-lm / vLLM); 0 = off")
    ap.add_argument("--rep-context", type=int, default=256, help="tokens the repetition penalty looks back over (mlx-lm)")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=None, help="override the sampling temperature (server backend); 0 = greedy")
    ap.add_argument("--no-thinking", action="store_true", help="ask with thinking off (server backend)")
    ap.add_argument("--top-p", type=float, default=None, help="override top-p (server backend)")
    ap.add_argument("--only-task", choices=["sci", "math", "code"], help="ask only this domain's questions")
    ap.add_argument("--grades-out", help="write the per-question grades here as JSON (for paired comparisons)")
    args = ap.parse_args()
    if args.quick:
        args.n_per_domain, args.max_tokens = 3, min(args.max_tokens, 4096)

    tasks_all = [json.loads(l) for l in open(args.tasks)]
    tasks = [t for t in tasks_all if t["order"] < args.n_per_domain]
    if args.only_task:
        tasks = [t for t in tasks if t["task"] == args.only_task]
    by_id = {t["id"]: t for t in tasks_all}
    if args.responses:
        rows = [json.loads(l) for l in open(args.responses)]
    else:
        done = {}
        part = Path(str(args.out) + ".partial")
        if args.resume:
            for src in (Path(args.out), part):
                if src.exists():
                    for l in open(src):
                        r = json.loads(l)
                        if not r.get("error") and r.get("text"):
                            done[r["id"]] = r
            print(f"resuming: {len(done)} answers kept", flush=True)
        todo = [t for t in tasks if t["id"] not in done]
        new = (ask_server(args, todo) if args.base_url else ask_transformers(args, todo)) if todo else []
        got = {**done, **{r["id"]: r for r in new if not r.get("error")}}
        rows = [got.get(t["id"]) or next(r for r in new if r["id"] == t["id"]) for t in tasks]
        failed = sum(1 for r in rows if r.get("error"))
        if failed:
            print(f"{failed} questions failed (server error); rerun with --resume to ask them again", flush=True)
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", args.out)
    ids = {r["id"] for r in rows}

    graded = {"yours": grade_rows(rows, by_id, args.allow_exec)}
    refs: dict[str, list] = {}
    if Path(args.refs).exists():
        for l in open(args.refs):
            r = json.loads(l)
            if r["id"] in ids:
                refs.setdefault(r.pop("model"), []).append(r)
    for name in ("orig", "q9b", "q4b", "pruned48", "lin_exp"):
        if name in refs:
            graded[name] = grade_rows(refs[name], by_id, args.allow_exec)
    if args.grades_out:
        Path(args.grades_out).write_text(json.dumps(graded, indent=0))
    table(graded)
    print("\nstrict = the benchmark's grader (needs </think> and \\boxed{}); lenient = also accepts answers without them;")
    print("loose = the last answer anywhere, thinking included (for answers that loop past the limit).")
    print("References were made with up to 16,000 tokens; compare like with like (--quick answers are cut at 4,096).")


if __name__ == "__main__":
    main()
