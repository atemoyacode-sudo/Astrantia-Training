"""Answers in other languages for heal_train.py (README 11, part L). Run on the Colab VM.

  prompts: people's own prompts from the Aya dataset (CohereForAI/aya_dataset) in the 23 non-English languages of
           wiki_text.py, plus English UltraChat prompts with a request to answer in one of those languages (phrased
           unlike the language probe and IFEval). Half are asked with thinking on, half off.
  traces:  the original's finished answers -> heal_train.py trace rows (domain "lang").

  python lang_data.py prompts --ultrachat /content/uc/train/data-00000-of-00019.arrow --out /content/lang
  python lang_data.py traces --out /content/lang      (after eval_chat.py wrote ask_on.answers.jsonl, ask_off.answers.jsonl)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LANGS = {  # wiki_text.py's list (by size and script), with Aya's ISO 639-3 codes
    "ja": ("Japanese", ["jpn"]), "zh": ("Chinese", ["zho"]), "ko": ("Korean", ["kor"]), "de": ("German", ["deu"]),
    "fr": ("French", ["fra"]), "es": ("Spanish", ["spa"]), "ru": ("Russian", ["rus"]), "it": ("Italian", ["ita"]),
    "pt": ("Portuguese", ["por"]), "nl": ("Dutch", ["nld"]), "pl": ("Polish", ["pol"]), "uk": ("Ukrainian", ["ukr"]),
    "ar": ("Arabic", ["arb", "ara"]), "fa": ("Persian", ["pes", "fas"]), "tr": ("Turkish", ["tur"]),
    "vi": ("Vietnamese", ["vie"]), "id": ("Indonesian", ["ind"]), "th": ("Thai", ["tha"]), "hi": ("Hindi", ["hin"]),
    "bn": ("Bengali", ["ben"]), "ta": ("Tamil", ["tam"]), "he": ("Hebrew", ["heb"]), "el": ("Greek", ["ell"]),
}
NATIVE, ASKED = 40, 15  # per language
FORMS = ["{p}\n\nRespond in {L}.", "{p}\n\nUse {L} for your reply.", "Could you answer this in {L}?\n\n{p}",
         "{p}\n\nYour answer should be written in {L}."]


def prompts(args) -> None:
    from datasets import load_dataset
    import pyarrow as pa

    rng = random.Random(3)
    aya = load_dataset("CohereForAI/aya_dataset", split="train")
    by = {}
    for r in aya:
        if 20 <= len(r["inputs"]) <= 1500:
            by.setdefault(r["language_code"], []).append(r["inputs"])
    with pa.memory_map(args.ultrachat) as src:
        convs = pa.ipc.open_stream(src).read_all().column("conversations").to_pylist()
    uc = [c[0]["content"] for c in convs[-4000:] if c and c[0]["role"] == "user" and 20 <= len(c[0]["content"]) <= 800]
    rng.shuffle(uc)
    rows, short = [], {}
    for code, (name, isos) in LANGS.items():
        pool = sorted({p for iso in isos for p in by.get(iso, [])})
        rng.shuffle(pool)
        native = pool[:NATIVE]
        if len(native) < NATIVE:
            short[code] = len(native)
        asked = [rng.choice(FORMS).format(p=uc.pop(), L=name) for _ in range(ASKED + NATIVE - len(native))]
        for i, p in enumerate(native + asked):
            rows.append({"task": "sci", "id": f"L-{code}-{'n' if i < len(native) else 'a'}{i}", "answer": "-",
                         "category": code, "prompt": p})
    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode, part in (("on", rows[::2]), ("off", rows[1::2])):
        with open(out / f"ask_{mode}.jsonl", "w") as f:
            for i, r in enumerate(part):
                f.write(json.dumps({**r, "order": i}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts ({sum('-n' in r['id'] for r in rows)} native); fewer Aya prompts than {NATIVE}: {short}")


def traces(args) -> None:
    out, n, kept = Path(args.out), 0, []
    for mode in ("on", "off"):
        asked = {r["id"]: r["prompt"] for r in map(json.loads, open(out / f"ask_{mode}.jsonl"))}
        for r in map(json.loads, open(out / f"ask_{mode}.answers.jsonl")):
            n += 1
            text = r.get("text") or ""
            if r.get("finish_reason") != "stop" or not text.strip():
                continue
            if mode == "on" and ("</think>" not in text or not text.split("</think>", 1)[1].strip()):
                continue
            if mode == "off" and "</think>" in text:
                continue
            kept.append({"id": r["id"], "domain": "lang", "prompt": asked[r["id"]], "completion": text, "finish_reason": "stop",
                         "thinking": mode == "on", "completion_tokens": int(r.get("completion_tokens") or len(text) // 3)})
    with open(out / "traces_lang.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(kept)} of {n} answers kept -> {out / 'traces_lang.jsonl'} "
          f"({sum(r['completion_tokens'] for r in kept):,} tokens)")
    for code in LANGS:
        print(code, sum(r["id"].startswith(f"L-{code}-") for r in kept), end="  ")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--ultrachat", required=True); p.add_argument("--out", required=True)
    t = sub.add_parser("traces"); t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prompts": prompts, "traces": traces}[args.cmd](args)


if __name__ == "__main__":
    main()
