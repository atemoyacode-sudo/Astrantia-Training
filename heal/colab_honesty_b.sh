#!/bin/bash
# Colab VM (G4), step B of README 8: cut the uneven student from the original (as heal/kaggle/build_uneven, on the
# GPU), fetch the UltraChat answers, check fla and the speed of the uneven experts.
#   bash heal/colab_honesty_b.sh
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs
pkill -f "bin/vllm serve" 2>/dev/null; sleep 5
pip install -q 'transformers==5.17.0' accelerate flash-linear-attention 2>&1 | tail -2
cd "$X/moe"
[ -f /content/student/g128b28160/config.json ] || python -u build_shape.py --source Qwen/Qwen3.5-35B-A3B --run "$IN" \
    --shapes g128b28160 --out /content/student --from-stats "$IN/stats.npz" --device cuda 2>&1 | grep --line-buffered -v 'falling back'
ls /content/student/g128b28160 | head -5; du -sh /content/student/g128b28160
[ -f /content/uc/train/data-00000-of-00019.arrow ] || hf download jiapingW/qwen3.5-35b-a3b-ultrachat-regen \
    train/data-00000-of-00019.arrow --repo-type dataset --local-dir /content/uc 2>&1 | tail -1
python - <<'P' || { echo "fla does not work here; using the torch path"; pip uninstall -y flash-linear-attention fla-core 2>&1 | tail -1; }
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f
B, T, H, K = 1, 256, 4, 128
q, k, v = (torch.randn(B, T, H, K, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32); b = torch.rand(B, T, H, device="cuda", dtype=torch.float16)
o, _ = f(q, k, v, g=g, beta=b, use_qk_l2norm_in_kernel=True); o.float().sum().backward(); print("fla ok", o.dtype)
P
python bench_uneven.py --device cuda --dtype float16 2>&1 | tail -8
echo "B_DONE $(date '+%H:%M')"
