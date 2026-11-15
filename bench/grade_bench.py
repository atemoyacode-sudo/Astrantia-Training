"""Grade run_bench.py outputs and compare runs question by question.

- sci  : the letter in the last \\boxed{} (the usual rule for these sets).
- math : the last \\boxed{} against the reference with math-verify (`pip install math-verify`); without it, a
         normalised string match (reported as such, and stricter).
- code : the last ```python``` block plus the HumanEval+ tests, run in a separate process with a time limit.
         This executes model-written code, so it refuses to run outside a disposable VM unless XMODEL_ALLOW_EXEC=1.

With thinking on, only the text after </think> counts; an answer cut off before </think> is wrong.

  python grade_bench.py /content/bench/orig.jsonl /content/bench/pruned48.jsonl      # second and later runs are compared with the first
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

try:
    from math_verify import parse, verify

    HAVE_MATH_VERIFY = True
except ImportError:
    HAVE_MATH_VERIFY = False


def answer_part(row: dict) -> str | None:
    text = row.get("text") or ""
    if row.get("thinking"):
        return text.split("</think>")[-1] if "</think>" in text else None
    return text


def last_boxed(text: str, raw: bool = False) -> str | None:
    i = text.rfind("\\boxed")
    while i != -1:
        j = text.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    inner = text[j + 1 : k] if raw else re.sub(r"\\[a-z]+|[{}\s,$]", "", text[j + 1 : k])
                    if inner.strip():
                        return inner
                    break
                k += 1
        i = text.rfind("\\boxed", 0, i)
    return None


def norm_math(s: str) -> str:
    s = re.sub(r"\\(left|right|!|,|;|:|displaystyle)", "", s)
    s = re.sub(r"\\[dt]?frac", r"\\frac", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("$", "").replace(" ", "")
    return s.rstrip(".")


def grade_sci(task, row):
    ans = answer_part(row)
    pred = last_boxed(ans) if ans else None
    m = re.search(r"[A-J]", pred.upper()) if pred else None
    return bool(m) and m.group() == task["answer"]


def grade_math(task, row):
    ans = answer_part(row)
    pred = last_boxed(ans, raw=True) if ans else None
    if pred is None:
        return False
    if HAVE_MATH_VERIFY:
        try:
            return bool(verify(parse(f"${task['answer']}$"), parse(f"${pred}$")))
        except Exception:
            pass
    return norm_math(pred) == norm_math(task["answer"])


def run_code(program: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, str(path)], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade_code(task, row, timeout=30.0):
    ans = answer_part(row)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", ans, re.S) if ans else []
    if not blocks:
        return False
    program = blocks[-1] + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n"
    return run_code(program, timeout)


GRADERS = {"sci": grade_sci, "math": grade_math, "code": grade_code}


def grade_file(path: str, tasks: dict) -> dict[str, bool]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if any(r["task"] == "code" for r in rows) and not (Path("/content").exists() or os.environ.get("XMODEL_ALLOW_EXEC") == "1"):
        raise SystemExit("code grading runs model-written code: do it on the Colab VM (or set XMODEL_ALLOW_EXEC=1)")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda r: GRADERS[r["task"]](tasks[r["id"]], r), rows))
    return {r["id"]: ok for r, ok in zip(rows, results)}


def summarize(name: str, correct: dict[str, bool], tasks: dict) -> None:
    for dom in ("sci", "math", "code"):
        ids = [i for i in correct if tasks[i]["task"] == dom]
        if ids:
            print(f"  {name:28s} {dom:5s} {np.mean([correct[i] for i in ids]) * 100:5.1f}%  (n={len(ids)})")


def paired(base: dict, other: dict, tasks: dict, n_boot: int = 5000) -> None:
    rng = np.random.default_rng(0)
    for dom in ("sci", "math", "code"):
        ids = sorted(i for i in base if i in other and tasks[i]["task"] == dom)
        if not ids:
            continue
        d = np.array([other[i] - base[i] for i in ids], dtype=float)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        print(f"    {dom:5s} diff {d.mean() * 100:+5.1f} pt  95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]  "
              f"(gained {int((d > 0).sum())}, lost {int((d < 0).sum())}, n={len(ids)})")


def main() -> None:
    tasks = {t["id"]: t for t in (json.loads(l) for l in open(HERE / "tasks.jsonl") if l.strip())}
    print("math grader:", "math-verify" if HAVE_MATH_VERIFY else "normalised string match (install math-verify)")
    graded = [(Path(p).stem, grade_file(p, tasks)) for p in sys.argv[1:]]
    for name, c in graded:
        summarize(name, c, tasks)
    base_name, base = graded[0]
    for name, c in graded[1:]:
        print(f"  {name} vs {base_name}:")
        paired(base, c, tasks)
    out = {name: c for name, c in graded}
    Path(sys.argv[1]).with_suffix(".graded.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
