#!/bin/bash
# Colab VM (G4), README 15: outside benchmarks, thinking on, up to 32,768 tokens, bf16 vLLM, no seed.
#   bash bench/colab_ext.sh R    the three reference models (NeoHorse-1-9B, Qwen3.5-9B, Qwen3.5-35B-A3B)
#   bash bench/colab_ext.sh C    cut the student, export run 3 and wikil-a (padded bf16), the same sets on both,
#                                then the usual 60 / set 2 / check set with the 8,192 cap for $CHOICE if it is set
# Answers: /content/ext_out/<name>/<set>.jsonl; grades: /content/ext_out/grades.json. A finished set is skipped on a
# restart (colab_serve_ask.sh's .done files).
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd); E=$X/cache/ext; O=/content/ext_out; IN=$X/cache/heal_inputs
step() { echo "== $1 $(date '+%H:%M')"; }
mkdir -p $O
python -c "import vllm" 2>/dev/null || { step "installing vllm"; pip install -q -U vllm 2>&1 | tail -1; }
pip install -q "transformers==5.20.2" 2>&1 | tail -1
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
if [ ! -f $E/lcb_tests.jsonl ]; then  # the Mac's task files are kept; only the tests are made here
  step "lcb tests"
  python $X/bench/ext_bench.py make --out /content/ext_made 2>&1 | tail -3
  cp /content/ext_made/lcb_tests.jsonl $E/
  python - <<P
import json
a = {json.loads(l)["id"] for l in open("$E/lcb.jsonl")}; b = {json.loads(l)["id"] for l in open("$E/lcb_tests.jsonl")}
print("lcb tasks", len(a), "with tests", len(a & b)); assert a <= b
P
fi
A="--max-tokens 32768 --timeout 30000"
ask() {  # ask <name> <model dir or repo>
  MAXLEN=36864 MNS=${MNS:-64} bash $X/bench/colab_serve_ask.sh "$1" "$2" $O aime $E/aime.jsonl "$A" mmlu $E/mmlu.jsonl "$A" lcb $E/lcb.jsonl "$A"
  (cd $X/bench && python ext_bench.py grade --dir $E --answers $O "$1" 2>&1 | tail -4 && cp $O/grades.json $O/grades_$1.json)
}
if [ "${1:-R}" = R ]; then
  ask qwen9b Qwen/Qwen3.5-9B
  ask neohorse TokenRhythm/NeoHorse-1-9B
  MNS=32 ask orig Qwen/Qwen3.5-35B-A3B
  echo "EXT_R_DONE $(date '+%H:%M')"
  exit 0
fi
# ---- C: our models
pkill -f "bin/vllm serve" 2>/dev/null; sleep 5
if [ ! -f /content/student/g128b28160/config.json ]; then
  step "student"
  pip install -q 'transformers==5.17.0' accelerate flash-linear-attention 2>&1 | tail -1
  (cd $X/moe && python -u build_shape.py --source Qwen/Qwen3.5-35B-A3B --run "$IN" --shapes g128b28160 --out /content/student \
     --from-stats "$IN/stats.npz" --device cuda 2>&1 | grep --line-buffered -v 'falling back' | tail -3)
fi
export_one() {  # export_one <name> <lora>
  local M=/content/models/$1
  [ -f $M/pad_check.json ] && return 0
  (cd $X/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u pad_uneven.py --student /content/student/g128b28160 \
     --lora $2 --run $IN --out $M --save-dtype bfloat16 --cal-prompts $X/cache/honesty/ask_things_on.jsonl --max-cal-prompts 120 \
     --check-prompts $X/fix_task/chat_eval_set.jsonl --no-check 2>&1 | grep --line-buffered -v -e 'falling back' -e 'Loading weights' -e MISSING | tail -3)
}
step "exports (two at once: each is CPU-bound on one core)"
pip install -q 'transformers==5.17.0' 2>&1 | tail -1
export_one run3 $IN/lora_run3.pt > $O/export_run3.log 2>&1 &
export_one wikila $IN/lora_wikila.pt > $O/export_wikila.log 2>&1 &
wait
pip install -q "transformers==5.20.2" 2>&1 | tail -1
for n in run3 wikila; do
  cat $O/export_$n.log | tail -3
  h=$(python -c "import json; print(json.load(open('/content/models/$n/pad_check.json'))['dummy_hits'])" 2>/dev/null || echo missing)
  echo "$n dummy hits: $h"
  [ "$h" = 0 ] && ask g128b_$n /content/models/$n || echo "$n not served (dummy hits $h)"
done
echo "EXT_C_DONE $(date '+%H:%M')"
