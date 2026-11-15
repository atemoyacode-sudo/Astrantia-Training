"""Does the model answer in the language it is asked for? A small probe of our own (not IFEval's prompts), README 10.

20 languages that each have their own script (so the check needs no language detector), 2 everyday requests each,
asked in English with "answer in <language>". An answer passes when at least 80% of its letters are in the
language's script (Japanese: Han + kana, with kana at least 10% of them; Chinese: Han with no kana).

  python lang_probe.py make --out ../cache/lang/lang.jsonl
  python lang_probe.py grade name=answers.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

LANGS = {  # code: (name, script test on one character)
    "ja": ("Japanese", None), "zh": ("Chinese (Simplified)", None), "ko": ("Korean", (0xAC00, 0xD7AF, 0x1100, 0x11FF, 0x3130, 0x318F)),
    "ru": ("Russian", (0x0400, 0x04FF)), "ar": ("Arabic", (0x0600, 0x06FF, 0x0750, 0x077F)), "hi": ("Hindi", (0x0900, 0x097F)),
    "bn": ("Bengali", (0x0980, 0x09FF)), "ta": ("Tamil", (0x0B80, 0x0BFF)), "te": ("Telugu", (0x0C00, 0x0C7F)),
    "kn": ("Kannada", (0x0C80, 0x0CFF)), "gu": ("Gujarati", (0x0A80, 0x0AFF)), "pa": ("Punjabi (Gurmukhi script)", (0x0A00, 0x0A7F)),
    "ml": ("Malayalam", (0x0D00, 0x0D7F)), "th": ("Thai", (0x0E00, 0x0E7F)), "he": ("Hebrew", (0x0590, 0x05FF)),
    "el": ("Greek", (0x0370, 0x03FF, 0x1F00, 0x1FFF)), "ka": ("Georgian", (0x10A0, 0x10FF)), "hy": ("Armenian", (0x0530, 0x058F)),
    "am": ("Amharic", (0x1200, 0x139F)), "si": ("Sinhala", (0x0D80, 0x0DFF)),
}
TOPICS = [
    "Explain why the sky is blue to a ten-year-old.",
    "Give three tips for sleeping better.",
    "Describe how a bicycle moves when you pedal.",
    "Write a short thank-you note to a teacher who helped you.",
    "Explain what a prime number is, with two examples.",
    "Summarize how the water cycle works.",
    "Suggest a plan for a relaxing weekend at home.",
    "Explain how to make a simple cup of tea.",
    "Describe two benefits of reading books every day.",
    "Explain what plants need to grow well.",
]
FORMS = ["{t} Answer in {L}.", "{t} Please write your whole reply in {L}.", "In {L}: {t}", "{t} (Reply only in {L}.)"]


def make(args) -> None:
    rows = []
    for i, (code, (name, _)) in enumerate(LANGS.items()):
        for j in range(2):
            t = TOPICS[(2 * i + j) % len(TOPICS)]
            rows.append({"task": "sci", "id": f"lang-{code}-{j}", "answer": "-", "category": code,
                         "prompt": FORMS[(i + j) % len(FORMS)].format(t=t, L=name), "order": len(rows)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} prompts -> {out}")


def in_ranges(c: str, ranges) -> bool:
    o = ord(c)
    return any(ranges[k] <= o <= ranges[k + 1] for k in range(0, len(ranges), 2))


HAN = (0x4E00, 0x9FFF, 0x3400, 0x4DBF, 0xF900, 0xFAFF)
KANA = (0x3040, 0x309F, 0x30A0, 0x30FF)


def passes(code: str, text: str) -> tuple[bool, float]:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if len(letters) < 20:
        return False, 0.0
    if code in ("ja", "zh"):
        han = sum(in_ranges(c, HAN) for c in letters)
        kana = sum(in_ranges(c, KANA) for c in letters)
        share = (han + kana) / len(letters)
        if code == "ja":
            return share >= 0.8 and kana >= 0.1 * (han + kana), share
        return share >= 0.8 and kana == 0, share
    share = sum(in_ranges(c, LANGS[code][1]) for c in letters) / len(letters)
    return share >= 0.8, share


def answer_of(r) -> str:
    text = r.get("text") or ""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return "" if "<think>" in text else text.strip()


def grade(args) -> None:
    for spec in args.answers:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        ok, fails = 0, []
        for r in rows:
            code = r["id"].split("-")[1]
            p, share = passes(code, answer_of(r))
            ok += p
            if not p:
                fails.append(f"{code}:{share:.2f}")
        print(f"{name:16s} {ok}/{len(rows)}   failed: {' '.join(sorted(fails))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--out", required=True)
    g = sub.add_parser("grade"); g.add_argument("answers", nargs="+")
    args = ap.parse_args()
    {"make": make, "grade": grade}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
