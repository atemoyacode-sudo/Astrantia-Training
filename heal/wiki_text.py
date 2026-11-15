"""Plain Wikipedia text for the W part of heal_train.py (--text). Run on the Colab VM (it streams wikimedia/wikipedia).

Articles are taken from the start, cut to --tokens tokens; an article whose first 64 tokens match a stored window
(results/xmodel_q35a3b: C, A, B, the Wikipedia windows KL and perplexity are measured on) is skipped, so the
measured windows stay unseen. Half of the pieces are English and Japanese (the languages the cut was measured in),
the rest are spread over the other large Wikipedias, a fixed list chosen by size and script, not by any test set.

  python wiki_text.py --tokenizer /content/student/g128b28160 --exclude ../../results/xmodel_q35a3b/data.npz \\
      --pieces 1600 --out /content/wiki_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

OTHER = ["zh", "ko", "de", "fr", "es", "ru", "it", "pt", "nl", "pl", "uk", "ar", "fa", "tr", "vi", "id", "th", "hi",
         "bn", "ta", "he", "el"]


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--pieces", type=int, default=1600)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ex = np.load(args.exclude)
    seen = {tuple(r[:64].tolist()) for k in ex.files for r in ex[k]}
    plan = {"en": args.pieces // 4, "ja": args.pieces // 4}
    rest = args.pieces - sum(plan.values())
    for i, lang in enumerate(OTHER):
        plan[lang] = rest // len(OTHER) + (i < rest % len(OTHER))
    n_out, skipped = 0, 0
    with open(args.out, "w") as f:
        for lang, want in plan.items():
            got = 0
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
            for row in ds.shuffle(seed=args.seed, buffer_size=10_000):
                ids = tok(row["text"], add_special_tokens=False)["input_ids"]
                if len(ids) < args.min_tokens:
                    continue
                if tuple(ids[:64]) in seen:
                    skipped += 1
                    continue
                seen.add(tuple(ids[:64]))
                f.write(json.dumps({"lang": lang, "title": row["title"], "text": tok.decode(ids[: args.tokens])},
                                   ensure_ascii=False) + "\n")
                got += 1
                if got >= want:
                    break
            n_out += got
            print(f"{lang}: {got}", flush=True)
    print(f"{n_out} pieces -> {args.out}; {skipped} skipped as measured windows", flush=True)
    os._exit(0)  # the streaming readers of `datasets` keep threads alive and the interpreter never exits


if __name__ == "__main__":
    main()
