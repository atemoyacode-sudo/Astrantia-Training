"""Questions for the teacher (the original Qwen3.5-35B-A3B) to answer with thinking on, for the light-healing run.

None of them come from the benchmark's sources (MMLU-Pro, MATH-500 / the MATH test split, HumanEval):
  math  MATH train split (levels 3-5) and GSM8K train, with the benchmark's math prompt
  sci   NVIDIA OpenScience, 10-option multiple choice (OS-Q2.5-72B-10), with the benchmark's science prompt
  code  MBPP (all splits) and Python tasks from Magicoder-OSS-Instruct-75K, with a prompt like the benchmark's

Any question sharing a 13-word sequence with a benchmark question (bench/tasks.jsonl, all 3 domains) is dropped.
The domains are interleaved (math, sci, code, math, ...) so that however far the teacher gets, the mix stays even.

  python xmodel/heal/build_prompts.py --per-domain 3000      # writes xmodel/heal/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench" / "tasks.jsonl"
HF = "https://huggingface.co/api/datasets"
SCI_PROMPT = ("Answer the following multiple choice question. Think step by step, then put the letter of the correct option "
              "(A-J) within \\boxed{{}}.\n\nQuestion: {q}\n\n{opts}")
MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
CODE_PROMPT = ("{task}\n\nReturn the complete solution (including any needed imports) in a single ```python``` code block.")
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
                 "precalculus"]


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


BOILERPLATE = [r"Answer the following multiple choice question\..*?within \\boxed\{\}\.", r"Please reason step by step, and put your final answer within \\boxed\{\}\.",
               r"Complete the following Python function\..*?code block\.", r"Return the complete solution .*? code block\."]


def ngrams(s: str, n: int = 13) -> set[tuple[str, ...]]:
    """Word 13-grams of the question itself (the prompt templates are removed first: they are shared on purpose)."""
    for b in BOILERPLATE:
        s = re.sub(b, " ", s, flags=re.S)
    w = words(s)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{") - 1
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1 : k]
    return None


def math_rows(rng) -> list[dict]:
    out = []
    for sub in MATH_SUBJECTS:
        df = pd.read_parquet(f"{HF}/EleutherAI/hendrycks_math/parquet/{sub}/train/0.parquet")
        for r in df.itertuples():
            if r.level in ("Level 3", "Level 4", "Level 5"):
                out.append({"domain": "math", "source": f"MATH-train/{sub}/{r.level[-1]}", "answer": boxed(r.solution),
                            "prompt": MATH_PROMPT.format(problem=r.problem)})
    gsm = pd.read_parquet(f"{HF}/openai/gsm8k/parquet/main/train/0.parquet")
    g = [{"domain": "math", "source": "gsm8k-train", "answer": r.answer.split("####")[-1].strip(),
          "prompt": MATH_PROMPT.format(problem=r.question)} for r in gsm.itertuples()]
    rng.shuffle(g)
    out += g[: len(out) // 4]  # about 1 GSM8K question in 5
    rng.shuffle(out)
    return out


def sci_rows(n: int, rng) -> list[dict]:
    """The first parquet file of OpenScience's 10-option config (about 40k questions; synthetic, mostly science)."""
    url = ("https://huggingface.co/datasets/nvidia/OpenScience/resolve/refs%2Fconvert%2Fparquet/OS-Q2.5-72B-10/"
           "partial-train/0000.parquet")
    df = pd.read_parquet(url, columns=["input", "output"])
    out = []
    for r in df.itertuples():
        text = r.input.strip()
        m = re.search(r"\n\s*A[:.)]\s", text)
        if not m:
            continue
        q, opts = text[: m.start()].strip(), text[m.start():].strip()
        lines = [re.sub(r"^\s*([A-J])[:.)]\s*", r"\1. ", ln) for ln in opts.splitlines() if ln.strip()]
        if len(lines) != 10 or not all(ln.startswith(f"{c}. ") for ln, c in zip(lines, "ABCDEFGHIJ")):
            continue
        out.append({"domain": "sci", "source": "OpenScience/OS-Q2.5-72B-10", "answer": boxed(r.output),
                    "prompt": SCI_PROMPT.format(q=q, opts="\n".join(lines))})
    rng.shuffle(out)
    return out


def code_rows(rng) -> list[dict]:
    out = []
    for split in ("train", "validation", "test", "prompt"):
        df = pd.read_parquet(f"{HF}/google-research-datasets/mbpp/parquet/full/{split}/0.parquet")
        for r in df.itertuples():
            task = f"{r.text}\nYour function should pass this test:\n{r.test_list[0]}"
            out.append({"domain": "code", "source": f"mbpp/{split}", "answer": None, "prompt": CODE_PROMPT.format(task=task)})
    mc = pd.read_parquet(f"{HF}/ise-uiuc/Magicoder-OSS-Instruct-75K/parquet/default/train/0.parquet")
    mc = mc[mc["lang"] == "python"]
    m = [{"domain": "code", "source": "magicoder-oss", "answer": None, "prompt": CODE_PROMPT.format(task=r.problem.strip())}
         for r in mc.itertuples() if len(r.problem) < 3000]
    out += m[: 3 * len(out)]  # about 1 MBPP task in 4, the rest Magicoder
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "prompts.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bench = set()
    for line in BENCH.open():
        bench |= ngrams(json.loads(line)["prompt"])
    pools = {"math": math_rows(rng), "sci": sci_rows(args.per_domain, rng), "code": code_rows(rng)}
    kept = {}
    for d, rows in pools.items():
        clean = [r for r in rows if not (ngrams(r["prompt"]) & bench)]
        print(f"{d}: {len(rows)} candidates, {len(rows) - len(clean)} dropped for 13-gram overlap with the benchmark")
        kept[d] = clean[: args.per_domain]
    rows = []
    for i in range(args.per_domain):
        for d in ("math", "sci", "code"):
            if i < len(kept[d]):
                rows.append({"id": f"{d}-{i:05d}", **kept[d][i]})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print({d: len(v) for d, v in kept.items()}, "->", args.out)


if __name__ == "__main__":
    main()
