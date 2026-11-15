#!/bin/bash
# On the Mac: one 4-bit MLX model (optionally with a LoRA adapter) answers a question set; answers are saved as they
# arrive and the server is restarted when mlx-lm runs out of Metal resources. Code is graded elsewhere (disposable VM).
#
#   bash heal/mac_eval.sh <out dir> [eval_chat.py arguments...]
#   MODEL=cache/mlx_96x256_q4 ADAPTER=cache/heal_adapters/6e6 DRY=1 CONC=3 bash heal/mac_eval.sh cache/x --tasks ...
set -euo pipefail
OUT=$1; shift
X=$(cd "$(dirname "$0")/.." && pwd)
PY=$X/.venv-mlx/bin/python
MODEL=${MODEL:-$X/cache/mlx_96x256_q4}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SERVER=("$X/heal/mlx_safe_server.py"); [ -n "${DRY:-}" ] && SERVER=("$X/heal/mlx_dry_server.py")
CACHE=(--prompt-cache-size 2 --prompt-cache-bytes 1073741824)  # every question is new: a big prompt cache only costs memory
AD=(); [ -n "${ADAPTER:-}" ] && AD=(--adapter-path "$ADAPTER")
echo "== start $OUT ($(date '+%m-%d %H:%M'))"
SRV=; trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT  # eval_chat.py failing must not leave the server running
for attempt in $(seq 1 30); do
  bash "$X/heal/mem_wait.sh"  # do not start a server while memory is short: the guard would kill it at once
  $PY "${SERVER[@]}" --model "$MODEL" ${AD[@]+"${AD[@]}"} --port "${PORT:-8080}" --max-tokens 16000 "${CACHE[@]}" >> "$OUT/server.log" 2>&1 &
  SRV=$!
  bash "$X/heal/mem_guard.sh" $SRV "${GUARD_GB:-18}" >> "$OUT/server.log" 2>&1 &
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:${PORT:-8080}/v1/models" > /dev/null && break; sleep 2; done
  (cd "$X/fix_task" && $PY -u eval_chat.py --base-url "http://127.0.0.1:${PORT:-8080}/v1" --model-name default_model \
     --no-seed --concurrency "${CONC:-3}" --resume --out "$OUT/answers.jsonl" "$@" 2>&1 | tee "$OUT/eval.txt")
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true  # the guard may have killed it already
  grep -q "questions failed" "$OUT/eval.txt" || break
  echo "== attempt $attempt left some unanswered; restarting the server ($(date +%H:%M))"
  sleep 5
done
# 14 Nov: g128b_wikil's usual 60 ran out of attempts with 24 still unanswered and was marked done anyway, so the
# queues skipped it and its error rows were graded as wrong answers. Say so and fail instead.
if grep -q "questions failed" "$OUT/eval.txt"; then
  echo "== gave up $OUT: $(grep -o '[0-9]* questions failed' "$OUT/eval.txt" | tail -1) after 30 attempts ($(date '+%m-%d %H:%M'))"
  exit 1
fi
echo "== done $OUT ($(date '+%m-%d %H:%M'))"
