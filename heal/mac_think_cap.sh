#!/bin/bash
# On the Mac: apply thinking caps (4,096 and 8,192 tokens) to stored 60-question answers of the cut model.
#   bash heal/mac_think_cap.sh <name> <answers.jsonl> [adapter dir]      -> cache/think_cap/<name>_cap<N>.jsonl
# env: MODEL, TASKS, CAPS ("4096 8192"), EXTRA (more think_cap.py flags, e.g. "--loop-cut --exit-note")
set -euo pipefail
NAME=$1; ANS=$2; ADAPTER=${3:-}
X=$(cd "$(dirname "$0")/.." && pwd)
PY=$X/.venv-mlx/bin/python
MODEL=${MODEL:-$X/cache/mlx_96x256_q4}
OUT=$X/cache/think_cap; mkdir -p "$OUT"
AD=(); [ -n "$ADAPTER" ] && AD=(--adapter-path "$ADAPTER")
# The mlx-lm server can die on a long loop ("generation thread died"); think_cap.py keeps finished answers in
# <out>.partial, so restart the server and go on until both cap files exist.
SRV=
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null || true' EXIT
for ATTEMPT in $(seq 1 ${ATTEMPTS:-8}); do
  MISSING=0
  for CAP in ${CAPS:-4096 8192}; do [ -f "$OUT/${NAME}_cap$CAP.jsonl" ] || MISSING=1; done
  [ $MISSING = 0 ] && break
  [ "$ATTEMPT" -gt 1 ] && echo "== cap attempt $ATTEMPT: restarting the server ($(date '+%H:%M'))"
  bash "$X/heal/mem_wait.sh"  # the guard kills a server started while memory is short (2026-11-13: 8 attempts lost in 7 minutes)
  $PY "$X/heal/mlx_safe_server.py" --model "$MODEL" ${AD[@]+"${AD[@]}"} --port 8080 --max-tokens 16000 \
      --prompt-cache-size 2 --prompt-cache-bytes 1073741824 >> "$OUT/server_$NAME.log" 2>&1 &
  SRV=$!
  bash "$X/heal/mem_guard.sh" $SRV "${GUARD_GB:-18}" >> "$OUT/server_$NAME.log" 2>&1 &
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 2; done
  for CAP in ${CAPS:-4096 8192}; do
    [ -f "$OUT/${NAME}_cap$CAP.jsonl" ] || $PY -u "$X/heal/think_cap.py" --answers "$ANS" --model "$MODEL" --cap "$CAP" \
        --tasks "${TASKS:-$X/fix_task/chat_eval_set.jsonl}" --concurrency "${CONC:-3}" ${EXTRA:-} --out "$OUT/${NAME}_cap$CAP.jsonl" 2>&1 | grep -v regex || break
  done
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=; sleep 5
done
for CAP in ${CAPS:-4096 8192}; do [ -f "$OUT/${NAME}_cap$CAP.jsonl" ] || { echo "cap $CAP not finished for $NAME"; exit 1; }; done
