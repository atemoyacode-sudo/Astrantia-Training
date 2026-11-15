# Training data (`astrantia_train.jsonl.gz`)

All the training rows written for this project, in one file: 5,278 rows, one JSON object per line, gzipped.
Every row is a prompt and the answer the student was trained to write (the original Qwen3.5-35B-A3B's own answer,
or an "I don't know" where step 3 says so).

| `stage` | `part` | rows | what | used by |
|---:|---|---:|---|---|
| 1 | `teacher` | 796 | the original's answers (thinking on) to math, science and code questions | healing step 1 (`heal/kaggle/train`) |
| 2 | `mix2` | 1,654 | the original's code / math / science answers, plus `honest` rows: real and made-up names, short facts, answered or declined as the original did | healing step 2 (`heal/colab_wiki_l.sh`) |
| 2 | `lang` | 1,020 | the original's answers in 23 languages (Aya prompts, and UltraChat prompts asking for another language) | healing step 2 |
| 3 | `mix3` | 1,808 | step 3's mix: facts the model knows, facts it repeats wrong (with the original's right answer), "I don't know" for the rest, with code / math / science answers | healing step 3 (`heal/colab_wikila.sh`) |

Fields: `stage`, `part`, `id`, `domain` (`math`, `sci`, `code`, `honest`, `lang`), `prompt`, `completion` (what is
trained on, from the start of the assistant turn), `finish_reason`, `completion_tokens`; and where present `source`
(the dataset the prompt came from), `answer` (the reference answer), `kind` (for `honest` rows), `thinking`
(whether thinking was on), `backend`, `prompt_tokens`.

Rows of stages 2 and 3 repeat some rows of earlier stages, as they were used: each stage's rows are the full input
of that stage.

## Not in this file

- **UltraChat answers** (30-50% of the tokens of every step): the original's answers to UltraChat, from the public
  dataset [`jiapingW/qwen3.5-35b-a3b-ultrachat-regen`](https://huggingface.co/datasets/jiapingW/qwen3.5-35b-a3b-ultrachat-regen)
  (first shard, first turn), read directly by `heal/heal_train.py`.
- **Wikipedia text** (40% of step 2's tokens): made by `heal/wiki_text.py` from `wikimedia/wikipedia` (seed 5, 1,600
  pieces of up to 2,048 tokens). Wikipedia is CC BY-SA 4.0.
- **The loop-exit phase** of step 1 is made from the `teacher` rows during training (`--loop-exit`).

## Sources and licenses

The answers were written by Qwen3.5-35B-A3B (Apache-2.0). The prompts come from:

| `source` | dataset | license |
|---|---|---|
| `MATH-train` | EleutherAI/hendrycks_math (train split) | MIT |
| `gsm8k-train` | openai/gsm8k (train split) | MIT |
| `OpenScience` | nvidia/OpenScience | see its dataset card |
| `mbpp` | google-research-datasets/mbpp | CC BY 4.0 |
| `magicoder-oss` | ise-uiuc/Magicoder-OSS-Instruct-75K | see its dataset card |
| `lang` rows | CohereForAI/aya_dataset; UltraChat prompts | Apache-2.0; see the UltraChat card |
| `honest` rows | names and facts written for this project and by the original | Apache-2.0 |

Check each source's terms before reusing rows outside research. No benchmark question is included on purpose: any
training question sharing a 13-word run with a benchmark question was dropped (`heal/build_prompts.py`).
