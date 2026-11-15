#!/bin/bash
# Colab VM (G4), README 13a (wikil-a): run 3's data and settings, started from the Wikipedia run's LoRA (wikil) instead
# of the 6M LoRA. Runs in the run 3 session after run 3 (the student is already cut). Only if units are left.
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs
[ -f /content/heal3a/lora_final.pt ] || OUT=/content/heal3a TRACES=$X/cache/honesty/traces_mix3.jsonl INIT=$IN/lora_wikil.pt \
  MILESTONES=1e6,2e6,1e9 MAX_HOURS=0.8 bash "$X/heal/colab_honesty_c.sh"
[ -f /content/heal3a/lora_final.pt ] || { echo "no LoRA from the training"; exit 1; }
echo "WIKILA_DONE $(date '+%H:%M')"
