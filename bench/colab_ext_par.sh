#!/bin/bash
# Colab VM (G4), README 15 (changed 17:05): like colab_ext2.sh "sets", but all sets are sent at once to one
# server, so the long answers of one set do not leave the GPU idle at its end. Same settings otherwise; the number of
# parallel requests does not change the answers.
#   MNS=64 bash bench/colab_ext_par.sh <name> <model> <set...>
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd); E=$X/cache/ext; O=/content/ext_out
n=$1; m=$2; shift 2; OUT=$O/$n; mkdir -p $OUT; MNS=${MNS:-64}
[ -n "$(echo "$@" | grep ifeval)" ] && pip install -q "lm-eval[ifeval]" langdetect immutabledict nltk 2>&1 | tail -1
pip install -q "transformers==5.20.2" 2>&1 | tail -1
pkill -f "bin/vllm serve" 2>/dev/null; sleep 3
vllm serve "$m" --served-model-name "$n" --port 8000 --max-model-len 36864 --gpu-memory-utilization 0.88 \
    --max-num-seqs $MNS > "$OUT/vllm.log" 2>&1 &
pid=$!
for i in $(seq 1 480); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
  kill -0 $pid 2>/dev/null || { echo "== $n: vLLM exited"; tail -25 "$OUT/vllm.log"; exit 1; }
  sleep 5
done
echo "== $n serving ($(date '+%H:%M')); $(grep -i 'maximum concurrency' $OUT/vllm.log | tail -1 | cut -c1-160)"
for s in "$@"; do
  [ -f "$OUT/$s.done" ] && continue
  echo "== $n $s ($(date '+%H:%M'))"
  ( (cd $X/fix_task && python -u eval_chat.py --base-url http://127.0.0.1:8000/v1 --model-name "$n" --no-seed --concurrency $MNS \
      --resume --refs none --tasks "$E/$s.jsonl" --out "$OUT/$s.jsonl" --max-tokens 32768 --timeout 30000) > "$OUT/$s.txt" 2>&1
    if grep -q "questions failed" "$OUT/$s.txt"; then echo "== $n $s left some unanswered"; else touch "$OUT/$s.done"; echo "== $n $s done ($(date '+%H:%M'))"; fi ) &
done
wait $(jobs -p | grep -v "^$pid$") 2>/dev/null
for s in "$@"; do while [ ! -f "$OUT/$s.done" ] && pgrep -f "eval_chat.py.*--out $OUT/$s.jsonl" > /dev/null; do sleep 10; done; done
kill $pid 2>/dev/null
(cd $X/bench && python ext_bench.py grade --dir $E --answers $O "$n" 2>&1 | tail -5 && cp $O/grades.json $O/grades_$n.json)
echo "SETS_DONE $n $(date '+%H:%M')"
