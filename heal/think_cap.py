"""A thinking cap applied to stored answers: if an answer thought for more than --cap tokens, cut the thinking there,
close </think>, and let the model write only the final answer. Answers that thought for less are kept as they are.

The first --cap tokens of thinking under a cap are distributed exactly like the first --cap tokens of an uncapped run
(same model, same sampler), so the stored answers can be reused and only the short final part is generated.

  python think_cap.py --answers answers_full.jsonl --model ../cache/mlx_96x256_q4 --cap 4096 \
      --base-url http://127.0.0.1:8080/v1 --out answers_cap4096.jsonl
  python ../fix_task/eval_chat.py --responses answers_cap4096.jsonl       # grade (code on a disposable VM)

--loop-cut: a watchdog on the thinking. The thinking is also cut where it starts to repeat (the first 4,000-character
window that compresses below 15% with zlib, the same rule the grader uses for "looping"), whichever comes first.
--exit-note adds one fixed sentence (EXIT_NOTE) to the thinking after such a cut, before </think>.
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
EXIT_NOTE = "I am going in circles. I will stop here and give my best answer now."


def loop_start(th: str, win: int = 4000, step: int = 200, thr: float = 0.15) -> int | None:
    """The start of the first window of the thinking that is a loop, or None."""
    for i in range(0, len(th) - win + 1, step):
        w = th[i:i + win].encode()
        if len(zlib.compress(w, 9)) / len(w) < thr:
            cut = th.rfind("\n", max(0, i - 300), i + 1)  # end on a line break when one is near
            return cut if cut > 0 else i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--model", required=True, help="the mlx model directory (for its tokenizer and chat template)")
    ap.add_argument("--tasks", default=str(HERE.parent / "fix_task" / "chat_eval_set.jsonl"))
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--answer-tokens", type=int, default=2048)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model-name", default="default_model")
    ap.add_argument("--no-think-tag", action="store_true", help="answers from vLLM: the text does not start with <think> "
                    "(15 Nov: without this, every G4 answer was taken as unthinking and none was capped)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--loop-cut", action="store_true", help="also cut the thinking where it starts to loop")
    ap.add_argument("--exit-note", action="store_true", help="put EXIT_NOTE after a loop cut, before </think>")
    ap.add_argument("--temperature", type=float, default=0.6, help="for the answer after the cap (match the run's sampler)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = {t["id"]: t["prompt"] for t in map(json.loads, open(args.tasks))}
    rows = [json.loads(l) for l in open(args.answers)]

    def one(r):
        text = r["text"]
        if text.startswith("<think>"):
            body = text[len("<think>"):]
        elif args.no_think_tag:  # vLLM: the prompt's "<think>\n" is not echoed, the text starts inside the thinking
            body = text
        else:  # the model closed its thinking at once: nothing to cap
            return {**r, "capped": False}
        thinking = body.split("</think>")[0]
        ids = tok(thinking, add_special_tokens=False)["input_ids"]
        at = loop_start(thinking) if args.loop_cut else None
        if at is not None and len(tok(thinking[:at], add_special_tokens=False)["input_ids"]) <= args.cap:
            cut, kept, why = thinking[:at].rstrip() + ("\n\n" + EXIT_NOTE if args.exit_note else ""), None, "loop"
            kept = len(tok(cut, add_special_tokens=False)["input_ids"])
        elif len(ids) <= args.cap:
            return {**r, "capped": False}
        else:
            cut, kept, why = tok.decode(ids[: args.cap]), args.cap, "cap"
        head = tok.apply_chat_template([{"role": "user", "content": prompts[r["id"]]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        assert head.endswith("<think>\n"), head[-40:]
        prompt = head + cut.lstrip("\n") + "\n</think>\n\n"
        body = {"model": args.model_name, "prompt": prompt, "max_tokens": args.answer_tokens, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": 20}
        for attempt in range(4):
            try:
                j = requests.post(f"{args.base_url}/completions", json=body, timeout=3600)
                j.raise_for_status()
                ch = j.json()["choices"][0]
                return {**r, "text": f"<think>{cut}\n</think>\n\n{ch['text']}", "finish_reason": ch.get("finish_reason"),
                        "completion_tokens": kept + j.json().get("usage", {}).get("completion_tokens", 0), "capped": True,
                        "cut_by": why}
            except Exception as e:
                err = repr(e)
        raise SystemExit(f"{r['id']}: {err}")

    part = Path(args.out + ".partial")  # finished answers are kept as they arrive, so a stopped run can go on
    done = {r["id"]: r for r in map(json.loads, open(part))} if part.exists() else {}
    lock = __import__("threading").Lock()

    def keep(r):
        # 14 Nov: a server-error row of the source was kept as "done" and stayed wrong after the source was answered;
        # only a kept row without an error, from a source row without one, is reused
        if r["id"] in done and not done[r["id"]].get("error"):
            return done[r["id"]]
        o = one(r)
        with lock, open(part, "a") as f:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
        return o

    with ThreadPoolExecutor(args.concurrency) as pool:
        out = list(pool.map(keep, rows))
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.unlink()
    loops = sum(r.get("cut_by") == "loop" for r in out)
    print(f"capped {sum(r['capped'] for r in out)}/{len(out)} answers at {args.cap} thinking tokens"
          f"{f' ({loops} at a loop)' if args.loop_cut else ''} -> {args.out}")


if __name__ == "__main__":
    main()
