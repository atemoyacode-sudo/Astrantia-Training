#!/bin/bash
# Colab VM (G4), README 9b: the exact benchmark of the current model (6M LoRA) in bf16 with vLLM, in the run 3 session
# after wikil-a (the student is already cut). The same steps as the 2x T4 kernel (heal/kaggle/bench_uneven_a): fold the
# LoRA into a padded checkpoint (pad_uneven.py), answer with seed 0 and 16k tokens, then the 8,192 cap of the same
# answers; then the second fresh set and the check set the same way. Grading runs here (code is executed on this VM).
# Each finished stage leaves its files in /content/bench, so a stop part-way keeps what is done.
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs; B=${B:-/content/bench}; M=${M:-/content/models/cur}; NAME=${NAME:-g128b_heal6e6}
LORA=${LORA:-$IN/lora_6e6.pt}  # env: README 15 runs it for the chosen model (its export already made by bench/colab_ext.sh C)
step() { echo "== $1 $(date '+%H:%M')"; }
mkdir -p $B
if [ ! -f $M/config.json ]; then
  step "export"
  (cd $X/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u pad_uneven.py --student /content/student/g128b28160 \
     --lora $LORA --run $IN --out $M --cal-prompts $X/cache/honesty/ask_things_on.jsonl --max-cal-prompts 120 \
     --check-prompts $X/fix_task/chat_eval_set.jsonl --no-check 2>&1 | grep --line-buffered -v -e 'falling back' -e 'Loading weights' -e MISSING | tail -6)
fi
cp $M/pad_check.json $B/ 2>/dev/null
python -c "import json,sys; h=json.load(open('$M/pad_check.json'))['dummy_hits']; print('dummy hits', h); sys.exit(1 if h else 0)" \
  || { echo "a dummy expert would be picked; stopping"; exit 1; }
step "serve"
# as run 1/2's guard on this VM: the training pins transformers 5.17, the vLLM installed first wants the newer one
pip install -q "transformers==5.20.2" 2>&1 | tail -1
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
pkill -f "bin/vllm serve" 2>/dev/null; sleep 3
vllm serve $M --served-model-name $NAME --port 8000 --max-model-len 20480 --gpu-memory-utilization 0.88 --max-num-seqs 64 \
  > $B/vllm.log 2>&1 &
pid=$!
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
  kill -0 $pid 2>/dev/null || { echo "vLLM exited"; tail -30 $B/vllm.log; exit 1; }
  sleep 5
done
step "serving"
one() {  # one <label> <tasks file> <refs: none or default>
  local label=$1 tasks=$2 refs=$3 r=(); [ "$refs" = none ] && r=(--refs none)
  if [ ! -f $B/$label.done ]; then
    step "$label"
    (cd $X/fix_task && python -u eval_chat.py --base-url http://127.0.0.1:8000/v1 --model-name $NAME --out $B/$label.jsonl \
       --tasks $tasks "${r[@]}" --concurrency 64 --timeout 30000 --resume --allow-exec --grades-out $B/${label}_grades.json) \
       > $B/${label}_eval.txt 2>&1
    grep -q "questions failed" $B/${label}_eval.txt || touch $B/$label.done
    grep -A4 "^model" $B/${label}_eval.txt | head -5
  fi
  if [ -f $B/$label.done ] && [ ! -f $B/${label}_cap8192.done ]; then
    step "$label cap 8192"
    (cd $X/heal && python -u think_cap.py --answers $B/$label.jsonl --model $M --cap 8192 --tasks $tasks \
       --base-url http://127.0.0.1:8000/v1 --model-name $NAME --no-think-tag --concurrency 16 --out $B/${label}_cap8192.jsonl 2>&1 | grep -v regex | tail -2)
    (cd $X/fix_task && python -u eval_chat.py --responses $B/${label}_cap8192.jsonl --tasks $tasks "${r[@]}" --allow-exec \
       --grades-out $B/${label}_cap8192_grades.json) > $B/${label}_cap8192_eval.txt 2>&1 && touch $B/${label}_cap8192.done
    grep -A4 "^model" $B/${label}_cap8192_eval.txt | head -5
  fi
}
one usual60 $X/fix_task/chat_eval_set.jsonl default
one set2 $X/fix_task/check_set2.jsonl none
one check $X/fix_task/check_set.jsonl none
kill $pid 2>/dev/null
echo "BENCH_DONE $(date '+%H:%M')"
