"""Outside benchmarks for the release check (README 9): AIME 2026, MMLU-Pro (all 14 categories), LiveCodeBench v6
(the 175 problems added in release 6) and IFEval. Run on a disposable VM only: `grade lcb` executes model-written code.

  python ext_bench.py make --out /content/ext [--mmlu-per-cat 20] [--aime-samples 3] [--exclude prompts.jsonl ...]
      -> aime.jsonl, mmlu.jsonl, lcb.jsonl, ifeval.jsonl (eval_chat.py task files) and lcb_tests.jsonl, ifeval_meta.jsonl
  python ext_bench.py grade --dir /content/ext --answers /content/ext_out name1 name2 ...
      reads <answers>/<name>/<bench>.jsonl and prints one table (also <answers>/grades.json)

Questions are asked with eval_chat.py (thinking on, the usual sampling, no seed). Scores: AIME = mean over the samples
of exact integer matches; MMLU-Pro = letter in the last \\boxed{}; LiveCodeBench = all public and private tests pass
(last ```python block of the answer); IFEval = all instructions of a prompt followed (strict) and per instruction.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
SCI_PROMPT = ("Answer the following multiple choice question. Think step by step, then put the letter of the correct option "
              "(A-J) within \\boxed{{}}.\n\nQuestion: {q}\n\n{opts}")
MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
LCB_HEAD = ("You will be given a question (problem specification) and will generate a correct Python program that matches "
            "the specification and passes all tests.\n\n### Question:\n{q}\n\n")
LCB_STARTER = ("### Format: You will use the following starter code to write the solution to the problem and enclose your "
               "code within delimiters.\n```python\n{starter}\n```\n\n### Answer: (use the provided format with backticks)\n\n")
LCB_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
             "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program "
             "runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
             "### Answer: (use the provided format with backticks)\n\n")


def parquet(repo: str, split: str):
    import pandas as pd

    return pd.read_parquet(f"https://huggingface.co/api/datasets/{repo}/parquet/default/{split}/0.parquet")


def row(task, rid, prompt, answer, category, n):
    return {"task": task, "id": rid, "answer": answer, "category": category, "prompt": prompt, "order": 0, "bench_order": n}


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{path.name}: {len(rows)}", flush=True)


def make(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = " ".join(json.loads(l).get("prompt", "") for p in args.exclude for l in open(p))
    # AIME 2026
    df = parquet("MathArena/aime_2026", "train")
    write(out / "aime.jsonl", [row("math", f"aime-{r.problem_idx}-s{k}", MATH_PROMPT.format(problem=r.problem), str(r.answer), "aime", n)
                               for n, (r, k) in enumerate((r, k) for r in df.itertuples() for k in range(args.aime_samples))])
    # MMLU-Pro: the same number from each category, none that this project has asked or trained on
    df = parquet("TIGER-Lab/MMLU-Pro", "test")
    rows, skipped = [], 0
    for cat in sorted(df["category"].unique()):
        part = list(df[df["category"] == cat].itertuples())
        random.Random(0).shuffle(part)
        kept = 0
        for r in part:
            if kept == args.mmlu_per_cat:
                break
            if r.question[:200] in seen:
                skipped += 1
                continue
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r.options))
            rows.append(row("sci", f"mmlupro-{r.question_id}", SCI_PROMPT.format(q=r.question, opts=opts), r.answer, cat, len(rows)))
            kept += 1
    print(f"MMLU-Pro: {skipped} questions skipped because this project has used them", flush=True)
    write(out / "mmlu.jsonl", rows)
    # LiveCodeBench release 6 (test6.jsonl holds the problems added in v6)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    rows, tests = [], []
    for line in open(path):
        p = json.loads(line)
        prompt = LCB_HEAD.format(q=p["question_content"]) + (LCB_STARTER.format(starter=p["starter_code"]) if p["starter_code"] else LCB_STDIN)
        rid = f"lcb-{p['question_id']}"
        rows.append(row("code", rid, prompt, "-", f"{p['platform']}/{p['difficulty']}", len(rows)))
        public = json.loads(p["public_test_cases"])
        try:
            private = json.loads(p["private_test_cases"])
        except Exception:
            private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(p["private_test_cases"].encode()))))
        meta = json.loads(p["metadata"]) if p.get("metadata") else {}
        tests.append({"id": rid, "func_name": meta.get("func_name"), "tests": public + private})
    write(out / "lcb.jsonl", rows)
    write(out / "lcb_tests.jsonl", tests)
    # IFEval
    df = parquet("google/IFEval", "train")
    rows, meta = [], []
    for r in df.itertuples():
        rows.append(row("sci", f"ifeval-{r.key}", r.prompt, "-", "ifeval", len(rows)))
        meta.append({"id": f"ifeval-{r.key}", "prompt": r.prompt, "instruction_id_list": list(r.instruction_id_list),
                     "kwargs": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(kw).items() if v is not None} for kw in r.kwargs]})
    write(out / "ifeval.jsonl", rows)
    write(out / "ifeval_meta.jsonl", meta)


# ---------------------------------------------------------------- grading


def answer_of(r) -> str:
    """The final answer: the text after </think>; an answer still inside its thinking has none; an answer given with
    thinking off (no <think> at all) is the whole text."""
    text = r.get("text") or ""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return "" if "<think>" in text else text.strip()


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + 6
    for j in range(i + 6, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            break
    return text[i + 7 : j].strip()


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def grade_aime(rows, tasks):
    gold = {t["id"]: t["answer"] for t in tasks}
    per = defaultdict(list)
    for r in rows:
        b = last_boxed(answer_of(r)) or ""
        digits = re.sub(r"[^\d]", "", re.sub(r"\\text\{[^}]*\}", "", b))
        per[r["id"].rsplit("-", 1)[0]].append(digits.lstrip("0") == gold[r["id"]].lstrip("0") and digits != "")
    score = sum(sum(v) / len(v) for v in per.values())
    return {"score": score, "of": len(per), "detail": f"{score:.1f}/{len(per)}"}


def grade_mmlu(rows, tasks):
    info = {t["id"]: t for t in tasks}
    cat = defaultdict(lambda: [0, 0])
    for r in rows:
        b = last_boxed(answer_of(r)) or ""
        m = re.search(r"[A-J]", re.sub(r"\\text\{([^}]*)\}", r"\1", b))
        t = info[r["id"]]
        cat[t["category"]][0] += bool(m) and m.group(0) == t["answer"]
        cat[t["category"]][1] += 1
    right, n = sum(v[0] for v in cat.values()), sum(v[1] for v in cat.values())
    return {"score": right, "of": n, "detail": f"{right}/{n}", "by_category": {k: v for k, v in sorted(cat.items())}}


def last_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def same_output(got: str, want: str) -> bool:
    g, w = [l.split() for l in got.strip().splitlines()], [l.split() for l in want.strip().splitlines()]
    if g == w:
        return True
    try:  # numbers that differ only in formatting
        return len(g) == len(w) and all(len(a) == len(b) and all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b)) for a, b in zip(g, w))
    except ValueError:
        return False


FUNC_HARNESS = '''
import json, sys
{code}
_args = [json.loads(l) for l in sys.stdin.read().strip().split("\\n")]
_out = getattr(Solution(), {name!r})(*_args)
print(json.dumps(_out))
'''


def run_one(code: str, test: dict, func_name: str | None, timeout: float) -> bool:
    functional = test.get("testtype") == "functional"
    src = FUNC_HARNESS.format(code="from typing import *\nimport collections, heapq, bisect, math, itertools, functools\n" + code, name=func_name) if functional else code
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "sol.py"
        f.write_text(src)
        try:
            p = subprocess.run([sys.executable, str(f)], input=test["input"], capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False
    if p.returncode != 0:
        return False
    if functional:
        try:
            return json.loads(p.stdout.strip().splitlines()[-1]) == json.loads(test["output"])
        except Exception:
            return False
    return same_output(p.stdout, test["output"])


def grade_lcb(rows, tasks, tests_path: Path, timeout: float):
    tests = {t["id"]: t for t in map(json.loads, open(tests_path))}
    info = {t["id"]: t for t in tasks}

    def one(r) -> bool:
        code = last_code(answer_of(r))
        t = tests[r["id"]]
        return bool(code) and all(run_one(code, x, t["func_name"], timeout) for x in t["tests"])

    with ThreadPoolExecutor(max(2, (os.cpu_count() or 4) - 2)) as ex:
        passed = list(ex.map(one, rows))
    by = defaultdict(lambda: [0, 0])
    for r, ok in zip(rows, passed):
        d = info[r["id"]]["category"].split("/")[-1]
        by[d][0] += ok
        by[d][1] += 1
    return {"score": sum(passed), "of": len(rows), "detail": f"{sum(passed)}/{len(rows)}", "by_category": dict(sorted(by.items()))}


def grade_ifeval(rows, meta_path: Path):
    try:  # Google's checker, as shipped in lm-evaluation-harness (pip install "lm-eval[ifeval]")
        from lm_eval.tasks.ifeval import instructions_registry
    except Exception:
        try:
            from instruction_following_eval import instructions_registry
        except Exception as e:  # without a checker the answers stay ungraded
            return {"score": None, "of": len(rows), "detail": f"no checker ({type(e).__name__})"}
    meta = {m["id"]: m for m in map(json.loads, open(meta_path))}
    strict, inst_ok, inst_n = 0, 0, 0
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        m, resp = meta[r["id"]], answer_of(r)
        oks = []
        for iid, kw in zip(m["instruction_id_list"], m["kwargs"]):
            by_kind.setdefault(iid, [0, 0])[1] += 1
            ins = instructions_registry.INSTRUCTION_DICT[iid](iid)
            ins.build_description(**{k: v for k, v in kw.items() if v is not None})
            args = ins.get_instruction_args()
            if args and "prompt" in args:
                ins.build_description(prompt=m["prompt"])
            try:
                oks.append(bool(resp.strip()) and bool(ins.check_following(resp)))
            except Exception:
                oks.append(False)
            by_kind[iid][0] += oks[-1]
        strict += all(oks)
        inst_ok += sum(oks)
        inst_n += len(oks)
    return {"score": strict, "of": len(rows), "detail": f"{strict}/{len(rows)} ({inst_ok}/{inst_n} instructions)",
            "by_category": dict(sorted(by_kind.items()))}


def grade(args) -> None:
    d, res = Path(args.dir), {}
    tasks = {b: load(d / f"{b}.jsonl") for b in ("aime", "mmlu", "lcb", "ifeval")}
    for name in args.names:
        a = Path(args.answers) / name
        res[name] = {}
        for b in ("aime", "mmlu", "lcb", "ifeval"):
            rows = load(a / f"{b}.jsonl")
            if not rows:
                continue
            g = (grade_aime(rows, tasks[b]) if b == "aime" else grade_mmlu(rows, tasks[b]) if b == "mmlu" else
                 grade_lcb(rows, tasks[b], d / "lcb_tests.jsonl", args.timeout) if b == "lcb" else grade_ifeval(rows, d / "ifeval_meta.jsonl"))
            g["answered"] = len(rows)
            g["no_answer"] = sum(not answer_of(r) for r in rows)
            res[name][b] = g
            print(f"{name:14s} {b:7s} {g['detail']:>28s}   answered {len(rows)}, no final answer {g['no_answer']}", flush=True)
        (Path(args.answers) / "grades.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "grade"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--answers")
    ap.add_argument("--mmlu-per-cat", type=int, default=20)
    ap.add_argument("--aime-samples", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()
    make(args) if args.cmd == "make" else grade(args)


if __name__ == "__main__":
    main()
