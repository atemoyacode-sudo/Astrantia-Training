#!/bin/bash
# Colab VM (G4), README 13 (run 3): cut the student and train on traces_mix3. Only 29 compute units were left
# (14 Nov 20:45), so the export and the guard are not run here; the guard runs on the Mac (README 8's own form).
#   bash heal/colab_run3.sh            (each step is skipped when its output exists, so a restart resumes)
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs
step() { echo "== $1 $(date '+%H:%M')"; }
# vLLM first, as in run 2 (its torch is the one the training then uses)
python -c "import vllm" 2>/dev/null || { step "installing vllm"; pip install -q -U vllm 2>&1 | tail -1; }
step "student"
bash "$X/heal/colab_honesty_b.sh" 2>&1 | grep -v "^layer" | tail -12
step "training"
[ -f /content/heal3/lora_final.pt ] || OUT=/content/heal3 TRACES=$X/cache/honesty/traces_mix3.jsonl MILESTONES=1e6,2e6,1e9 \
  MAX_HOURS=1.2 bash "$X/heal/colab_honesty_c.sh"
[ -f /content/heal3/lora_final.pt ] || { echo "no LoRA from the training"; exit 1; }
echo "RUN3_DONE $(date '+%H:%M')"
