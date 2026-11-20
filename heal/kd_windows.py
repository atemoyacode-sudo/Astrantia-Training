"""README 21: 1,024-token windows for token-level KD, packed from three sources (a third each):

  chat       the original's own UltraChat answers (jiapingW/qwen3.5-35b-a3b-ultrachat-regen, first turn, prompt and
             answer in the chat template, thinking included)
  reasoning  the original's math, science and code answers (rows of data/astrantia_train.jsonl.gz or trace files)
  general    plain Wikipedia text (wiki_text.py's jsonl)

Within a source, examples are concatenated and cut into windows. A window sharing a 32-token run with any held-out
window (every array of the --exclude files) is dropped.

  python kd_windows.py --tokenizer Qwen/Qwen3.5-35B-A3B --ultrachat uc.arrow --traces astrantia_train.jsonl.gz \\
      --wiki wiki_text.jsonl --exclude data.npz data_general.npz --n 2000 --out kd_windows.npz
"""

from __future__ import annotations

import argparse
import gzip
import json
import random

import numpy as np

SOURCES = ("chat", "reasoning", "general")


def clean(text: str) -> str | None:
    text = (text or "").removeprefix("<think>\n").removeprefix("<think>")
    if "</think>" not in text or not text.split("</think>", 1)[1].strip():
        return None
    return text


def chat_ids(tok, prompt: str, completion: str) -> list[int]:
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    return tok(prefix + completion + "<|im_end|>\n", add_special_tokens=False)["input_ids"]


def ja_ids(tok, prompt: str, completion: str) -> list[int]:
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    return tok(prefix + completion + "<|im_end|>\n", add_special_tokens=False)["input_ids"]


def pack(streams, length: int, n: int, banned: set, k: int) -> list[list[int]]:
    """Concatenate token lists and cut into windows; skip windows with a banned k-gram."""
    out, buf = [], []
    for ids in streams:
        buf += ids
        while len(buf) >= length:
            w, buf = buf[:length], buf[length:]
            if not any(tuple(w[i : i + k]) in banned for i in range(0, length - k + 1)):
                out.append(w)
            if len(out) >= n:
                return out
    return out


def pack_whole(units, length: int, n: int, fill, banned: set, k: int) -> list[list[int]]:
    """README 35: whole examples (each at most `length`) placed in windows without cutting them, first fit in order,
    cycling over the units until n windows; the space left in a window is filled from the `fill` token stream."""
    out, buf, i = [], [], 0
    units = [u for u in units if len(u) <= length]
    while len(out) < n and units:
        w = []
        while len(w) + len(units[i % len(units)]) <= length:
            w += units[i % len(units)]
            i += 1
        while len(w) < length:
            if not buf:
                buf = next(fill)
            take = min(length - len(w), len(buf))
            w, buf = w + buf[:take], buf[take:]
        if not any(tuple(w[j : j + k]) in banned for j in range(0, length - k + 1)):
            out.append(w)
    return out


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ultrachat", required=True)
    ap.add_argument("--traces", nargs="+", required=True, help="jsonl(.gz) rows with prompt, completion (and domain)")
    ap.add_argument("--wiki", required=True, help="wiki_text.py output: jsonl of {text}")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--ja", default=None, help="README 27: ja_data.py rows.jsonl (thinking off), a fourth source")
    ap.add_argument("--ja-share", type=float, default=0.4, help="share of the windows from --ja")
    ap.add_argument("--pairs", default=None, help="README 25: pair_data.py rows.jsonl (thinking on), another source")
    ap.add_argument("--pairs-share", type=float, default=0.5, help="share of the windows from --pairs")
    ap.add_argument("--pairs-whole", action="store_true",
                    help="README 35: never cut a --pairs row; rows with think false use the thinking-off template")
    ap.add_argument("--n", type=int, default=2000, help="windows in all")
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--kgram", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    banned = set()
    for f in args.exclude:
        d = np.load(f)
        for key in d.files:
            for row in d[key]:
                row = row.tolist()
                banned.update(tuple(row[i : i + args.kgram]) for i in range(len(row) - args.kgram + 1))
    print(f"{len(banned):,} banned {args.kgram}-grams", flush=True)
    n_ja = int(args.n * args.ja_share) if args.ja else 0
    n_pairs = int(args.n * args.pairs_share) if args.pairs else 0
    per = (args.n - n_ja - n_pairs) // 3

    import pyarrow as pa
    with pa.memory_map(args.ultrachat) as src:
        convs = pa.ipc.open_stream(src).read_all().column("conversations").to_pylist()
    rng.shuffle(convs)

    def chat_stream():
        for conv in convs:
            if len(conv) >= 2 and conv[0]["role"] == "user" and conv[1]["role"] == "assistant":
                c = clean(conv[1]["content"])
                if c:
                    yield chat_ids(tok, conv[0]["content"], c)

    rows = []
    for f in args.traces:
        op = gzip.open if f.endswith(".gz") else open
        for line in op(f, "rt"):
            r = json.loads(line)
            if r.get("domain") in ("math", "sci", "code") and r.get("finish_reason") in (None, "stop"):
                rows.append(r)
    rng.shuffle(rows)
    seen = set()

    def reasoning_stream():
        for r in rows:
            key = (r["prompt"], r["completion"][:200])
            if key in seen:
                continue
            seen.add(key)
            c = clean(r["completion"])
            if c:
                yield chat_ids(tok, r["prompt"], c)

    texts = [json.loads(line)["text"] for line in open(args.wiki)]
    rng.shuffle(texts)

    def general_stream():
        for t in texts:
            yield tok(t, add_special_tokens=False)["input_ids"]

    def ja_stream():  # README 27: the original's answers with thinking off, in the template's no-thinking form
        ja_rows = [json.loads(line) for line in open(args.ja)]
        rng.shuffle(ja_rows)
        for r in ja_rows:
            yield ja_ids(tok, r["prompt"], r["completion"])

    wins, src = [], []
    plan = [(n, s, per) for n, s in zip(SOURCES, (chat_stream(), reasoning_stream(), general_stream()))]
    if args.ja:
        plan.append(("ja", ja_stream(), n_ja))
    if args.pairs:
        def pair_stream():  # README 25: kept pairs, the original's answers with its thinking
            prs = [json.loads(line) for line in open(args.pairs)]
            if prs and "group" in prs[0]:  # README 31: the two members of a pair stay next to each other
                groups = {}
                for r in prs:
                    groups.setdefault(r["group"], []).append(r)
                order = list(groups)
                rng.shuffle(order)
                prs = [r for g in order for r in groups[g]]
            else:
                rng.shuffle(prs)
            for r in prs:
                if r.get("think") is False:
                    yield ja_ids(tok, r["prompt"], r["completion"])
                    continue
                c = clean(r["completion"])
                if c:
                    yield chat_ids(tok, r["prompt"], c)

        if args.pairs_whole:
            plan.append(("pairs", None, n_pairs))
        else:
            plan.append(("pairs", pair_stream(), n_pairs))
    for i, (name, stream, want) in enumerate(plan):
        if stream is None:  # README 35: whole pairs, the gaps filled with more of the chat source
            w = pack_whole(list(pair_stream()), args.len, want, chat_stream(), banned, args.kgram)
        else:
            w = pack(stream, args.len, want, banned, args.kgram)
        print(f"{name}: {len(w)} windows", flush=True)
        wins += w
        src += [i] * len(w)
    order = list(range(len(wins)))
    rng.shuffle(order)
    np.savez(args.out, ids=np.array([wins[i] for i in order], dtype=np.int32), src=np.array([src[i] for i in order], dtype=np.int8))
    print(f"wrote {args.out}: {len(wins)} windows of {args.len}", flush=True)


if __name__ == "__main__":
    main()
