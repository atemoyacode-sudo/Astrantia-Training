#!/bin/bash
# On a Colab VM (G4, 96 GB): the original 35B-A3B in bf16 answers the fact probes, the notes and agreement tests and
# the two check sets (README 7, first step of stage 3). Nothing is graded here and no answer is executed: the files in
# $OUT are graded on the Mac (facts) and on a Kaggle CPU kernel (code). Answered sets are skipped on a second run.
#
#   pip install -U vllm   # once
#   OUT=/content/teacher_probe bash heal/colab_teacher.sh            # everything
#   OUT=/content/teacher_probe SETS="halluc halluc2" bash heal/colab_teacher.sh
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-/content/teacher_probe}
SETS=${SETS:-halluc halluc2 open rep check check2}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=8000
mkdir -p "$OUT"
# --max-num-seqs 64: with the default the store for the GDN state is too small and vLLM does not start (2026-11-08)
vllm serve "$MODEL" --served-model-name orig --port $PORT --max-model-len 20480 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 > "$OUT/vllm.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null' EXIT
for i in $(seq 1 480); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  kill -0 $pid 2>/dev/null || { echo "vLLM exited:"; tail -30 "$OUT/vllm.log"; exit 1; }
  sleep 5
done
echo "== serving $MODEL ($(date '+%H:%M'))"
ask() {  # name tasks [eval_chat.py arguments...]; no seed, as on the Mac (with a seed the five samples would be one)
  local name=$1 tasks=$2; shift 2
  [ -f "$OUT/$name.done" ] && { echo "== $name already answered"; return 0; }
  echo "== $name ($(date '+%H:%M'))"
  (cd "$X/fix_task" && python -u eval_chat.py --base-url "http://127.0.0.1:$PORT/v1" --model-name orig --no-seed \
     --concurrency 32 --resume --refs none --tasks "$X/fix_task/$tasks" --out "$OUT/$name.jsonl" "$@") > "$OUT/$name.txt" 2>&1
  if grep -q "questions failed" "$OUT/$name.txt"; then echo "== $name left some unanswered"; else touch "$OUT/$name.done"; fi
  tail -2 "$OUT/$name.txt"
}
for s in $SETS; do
  case $s in
    halluc)  ask halluc halluc_set.jsonl --max-tokens 1024 --no-thinking ;;   # 60 facts, 40 non-existent things, 10 known
    halluc2) ask halluc2 halluc_set2.jsonl --max-tokens 1024 --no-thinking ;; # the second 60 facts
    open)    ask open halluc_open.jsonl --max-tokens 256 --no-thinking ;;     # notes in the prompt (6m)
    rep)     ask rep halluc_rep.jsonl --max-tokens 256 --no-thinking ;;       # five samples per fact (6m)
    check)   ask check check_set.jsonl ;;                                     # thinking on, up to 16,000 tokens
    check2)  ask check2 check_set2.jsonl ;;
    *) echo "unknown set $s" ;;
  esac
done
echo "ALL_DONE $(date '+%H:%M')"
