#!/bin/bash
# Colab VM: serve one model with vLLM and ask it several task files with eval_chat.py (no grading, nothing executed).
#   bash bench/colab_serve_ask.sh <name> <model dir or repo> <out dir> <label> <tasks.jsonl> "<eval_chat.py args>" [more triples]
# Only the server process is stopped first ("bin/vllm serve"): a plain "vllm serve" pattern also matches any job
# whose command line names it, and killed run 2's first training start (2026-11-14).
# MAXLEN / MNS (env) set the context length and the number of parallel requests (default 20480 / 64).
# Answers go to <out dir>/<name>/<label>.jsonl; a set that is already answered is skipped.
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; MODEL=$2; OUT=$3/$1; shift 3
PORT=8000
mkdir -p "$OUT"
todo=0; args=("$@")
for ((i = 0; i < ${#args[@]}; i += 3)); do [ -f "$OUT/${args[i]}.done" ] || todo=1; done
[ $todo = 0 ] && { echo "== $NAME: all sets already answered"; exit 0; }
pkill -f "bin/vllm serve" 2>/dev/null; sleep 3
vllm serve "$MODEL" --served-model-name "$NAME" --port $PORT --max-model-len ${MAXLEN:-20480} --gpu-memory-utilization 0.88 \
    --max-num-seqs ${MNS:-64} > "$OUT/vllm.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null' EXIT
for i in $(seq 1 480); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  kill -0 $pid 2>/dev/null || { echo "== $NAME: vLLM exited"; tail -25 "$OUT/vllm.log"; exit 1; }
  sleep 5
done
echo "== $NAME serving ($(date '+%H:%M'))"
for ((i = 0; i < ${#args[@]}; i += 3)); do
  label=${args[i]}; tasks=$(realpath "${args[i+1]}"); extra=${args[i+2]}
  [ -f "$OUT/$label.done" ] && continue
  echo "== $NAME $label ($(date '+%H:%M'))"
  (cd "$X/fix_task" && python -u eval_chat.py --base-url "http://127.0.0.1:$PORT/v1" --model-name "$NAME" --no-seed \
     --concurrency ${MNS:-64} --resume --refs none --tasks "$tasks" --out "$OUT/$label.jsonl" $extra) > "$OUT/$label.txt" 2>&1
  if grep -q "questions failed" "$OUT/$label.txt"; then echo "== $NAME $label left some unanswered"; else touch "$OUT/$label.done"; fi
done
echo "== $NAME done ($(date '+%H:%M'))"
