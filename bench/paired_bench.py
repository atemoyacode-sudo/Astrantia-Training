"""Per-question comparison of two answer sets from ext_bench (same questions): right in A only / B only, the sign
test, and how many answers hit the token limit in each. LCB runs the model's code, so run this on a disposable VM.

  python paired_bench.py --dir ~/xmodel/cache/ext --a ~/p2/ext_out/p2 --b ~/wikila
"""
import argparse
import json
from math import comb
from pathlib import Path

import ext_bench as eb


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def per_q(b, rows, tasks, d):
    out = {}
    for r in rows:
        if b == "aime":
            g = eb.grade_aime([r], tasks)["score"]
        elif b == "mmlu":
            g = eb.grade_mmlu([r], tasks)["score"]
        elif b == "lcb":
            g = eb.grade_lcb([r], tasks, d / "lcb_tests.jsonl", 10.0)["score"]
        else:
            g = eb.grade_ifeval([r], d / "ifeval_meta.jsonl")["score"]
        out.setdefault(str(r["id"]), []).append(float(g))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    d, res = Path(args.dir), {}
    for b in ("aime", "mmlu", "lcb", "ifeval"):
        tasks = eb.load(d / f"{b}.jsonl")
        ra, rb = eb.load(Path(args.a) / f"{b}.jsonl"), eb.load(Path(args.b) / f"{b}.jsonl")
        if not ra or not rb:
            continue
        qa, qb = per_q(b, ra, tasks, d), per_q(b, rb, tasks, d)
        ids = sorted(set(qa) & set(qb))
        a_only = sum(qa[i] > qb[i] for i in ids)
        b_only = sum(qb[i] > qa[i] for i in ids)
        cut = lambda rows: sum(r.get("finish_reason") == "length" for r in rows)
        res[b] = {"n": len(ids), "a": sum(qa[i] for i in ids), "b": sum(qb[i] for i in ids), "a_better": a_only,
                  "b_better": b_only, "p": sign_p(a_only, b_only), "a_cut": cut(ra), "b_cut": cut(rb), "rows": len(ra)}
        print(b, json.dumps(res[b]), flush=True)
    (Path(args.a) / "paired.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
