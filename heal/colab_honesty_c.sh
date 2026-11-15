#!/bin/bash
# Colab VM (G4), step C of README 8: LoRA on the uneven student, started from the 6M LoRA, on the mixed answers.
#   bash heal/colab_honesty_c.sh            (SMOKE=1: three steps into /content/heal_smoke)
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs
cd "$X/moe" && cp ../heal/heal_train.py .
OUT=${OUT:-/content/heal}; EXTRA=""
[ -n "${SMOKE:-}" ] && { OUT=${OUT}_smoke; EXTRA="--smoke-steps 3"; }
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u heal_train.py --student /content/student/g128b28160 \
  --ultrachat /content/uc/train/data-00000-of-00019.arrow --run "$IN" --out $OUT \
  --traces "${TRACES:-$X/cache/honesty/traces_mix.jsonl}" --mix-traces --init-lora "${INIT:-$IN/lora_6e6.pt}" \
  --milestones "${MILESTONES:-5e5,1e6,1e9}" --max-hours "${MAX_HOURS:-1.0}" --g-share 0.3 --max-start-kl 0.3 $EXTRA 2>&1 \
  | grep --line-buffered -v 'falling back'
ls -la $OUT
echo "C_DONE $(date '+%H:%M')"
