#!/bin/bash
# Colab VM (G4), README 15 as changed at 14:35. Steps (each skips what is done):
#   bash bench/colab_ext2.sh prep               the student and the padded exports of run 3 and wikil-a (two at once)
#   bash bench/colab_ext2.sh sets <name> <model> <set...>   serve, ask the sets (aime mmlu lcb ifeval), grade
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd); E=$X/cache/ext; O=/content/ext_out; IN=$X/cache/heal_inputs
step() { echo "== $1 $(date '+%H:%M')"; }
mkdir -p $O
A="--max-tokens 32768 --timeout 30000"
if [ "$1" = prep ]; then
  pkill -f "bin/vllm serve" 2>/dev/null; sleep 5
  pip install -q 'transformers==5.17.0' accelerate flash-linear-attention 2>&1 | tail -1
  if [ ! -f /content/student/g128b28160/config.json ]; then
    step "student"
    (cd $X/moe && python -u build_shape.py --source Qwen/Qwen3.5-35B-A3B --run "$IN" --shapes g128b28160 --out /content/student \
       --from-stats "$IN/stats.npz" --device cuda 2>&1 | grep --line-buffered -v 'falling back' | tail -3)
  fi
  ex() {
    [ -f /content/models/$1/pad_check.json ] && return 0
    (cd $X/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u pad_uneven.py --student /content/student/g128b28160 \
       --lora $IN/lora_$1.pt --run $IN --out /content/models/$1 --save-dtype bfloat16 --cal-prompts $X/cache/honesty/ask_things_on.jsonl \
       --max-cal-prompts 120 --check-prompts $X/fix_task/chat_eval_set.jsonl --no-check 2>&1 | grep --line-buffered -v -e 'falling back' -e 'Loading weights' -e MISSING | tail -3)
  }
  step "exports"
  ex run3 > $O/export_run3.log 2>&1 &
  ex wikila > $O/export_wikila.log 2>&1 &
  wait
  for n in run3 wikila; do tail -2 $O/export_$n.log; echo "$n dummy hits: $(python -c "import json; print(json.load(open('/content/models/$n/pad_check.json'))['dummy_hits'])" 2>/dev/null || echo missing)"; done
  echo "PREP_DONE $(date '+%H:%M')"
  exit 0
fi
# sets <name> <model> <set...>
n=$2; m=$3; shift 3
if [[ "$m" == /content/models/* ]]; then
  h=$(python -c "import json; print(json.load(open('$m/pad_check.json'))['dummy_hits'])" 2>/dev/null || echo missing)
  [ "$h" = 0 ] || { echo "$n: dummy hits $h, not served"; echo "SETS_DONE $n $(date '+%H:%M')"; exit 0; }
fi
[ -n "$(echo "$@" | grep ifeval)" ] && pip install -q "lm-eval[ifeval]" langdetect immutabledict nltk 2>&1 | tail -1
pip install -q "transformers==5.20.2" 2>&1 | tail -1
triples=(); for s in "$@"; do triples+=("$s" "$E/$s.jsonl" "$A"); done
[ "$n" = orig ] && mns=32 || mns=64
MAXLEN=36864 MNS=$mns bash $X/bench/colab_serve_ask.sh "$n" "$m" $O "${triples[@]}"
(cd $X/bench && python ext_bench.py grade --dir $E --answers $O "$n" 2>&1 | tail -5 && cp $O/grades.json $O/grades_$n.json)
echo "SETS_DONE $n $(date '+%H:%M')"
