#!/bin/bash
# On the Mac: build a same-budget shape of Qwen3.5-35B-A3B (default 96x256, picked on chat data), convert it to 4-bit
# MLX, check its KL on the held-out chat windows, then let it answer the quick chat set (3 per domain, 4,096 tokens).
#
#   bash mac_shape.sh [shape] [quick|full]
#
# The build reads the whole original over HTTP (~67 GB; about an hour at ~25 MB/s). Code answers are not run here
# (grade them on a disposable VM); science and math are graded.
set -euo pipefail
SHAPE=${1:-96x256}; MODE=${2:-quick}
X=$(cd "$(dirname "$0")" && pwd)
PY=$X/.venv-mlx/bin/python
RUN=$X/../results/xmodel_q35a3b/chat_run
C=$X/cache
OUT=$C/mac_$SHAPE
mkdir -p "$OUT"
cd "$X/moe"
if [ ! -f "$C/mlx_${SHAPE}_q4/config.json" ]; then
  if [ ! -f "$C/shape/$SHAPE/config.json" ]; then
    echo "== build $SHAPE ($(date +%H:%M))"
    $PY -u build_shape.py --source Qwen/Qwen3.5-35B-A3B --run "$RUN" --shapes "$SHAPE" --out "$C/shape" --device mps \
      2>&1 | grep --line-buffered -v "falling back"
  fi
  echo "== 4-bit MLX ($(date +%H:%M))"
  $PY -u to_mlx.py "$C/shape/$SHAPE" "$C/mlx_${SHAPE}_q4" --q-bits 4 2>&1 | tail -3
fi
if [ ! -f "$C/mlx_${SHAPE}_q4/kl_chat.json" ]; then
  echo "== KL on held-out chat windows ($(date +%H:%M))"
  $PY -u mlx_kl.py --run "$RUN" --model "$C/mlx_${SHAPE}_q4" 2>&1 | tail -1
fi
echo "== answering ($MODE, $(date +%H:%M))"
EXTRA=""; [ "$MODE" = quick ] && EXTRA="--quick"
# mlx-lm can run out of Metal resources with many long answers at once ("generation thread died"); every answer is
# saved as it arrives, so restart the server and ask only the missing ones, up to 8 times.
for attempt in 1 2 3 4 5 6 7 8; do
  $PY -m mlx_lm server --model "$C/mlx_${SHAPE}_q4" --port 8080 --max-tokens 16000 >> "$OUT/server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 2; done
  (cd "$X/fix_task" && $PY -u eval_chat.py --base-url http://127.0.0.1:8080/v1 --model-name default_model --no-seed \
     --concurrency "${CONC:-4}" $EXTRA --resume --out "$OUT/answers_$MODE.jsonl" 2>&1 | tee "$OUT/eval_$MODE.txt")
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  grep -q "questions failed" "$OUT/eval_$MODE.txt" || break
  echo "== attempt $attempt left some unanswered; restarting the server ($(date +%H:%M))"
  sleep 5
done
echo "== done ($(date +%H:%M))"
