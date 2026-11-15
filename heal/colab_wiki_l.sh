#!/bin/bash
# Colab VM (G4), README 11 with part L: the original answers prompts in 23 languages (vLLM, bf16), then one run on
# run 2's data + those answers + plain Wikipedia text. Run after colab_wiki.sh has cut the student and built the text.
#   bash heal/colab_wiki_l.sh
set -uo pipefail
X=$(cd "$(dirname "$0")/.." && pwd)
IN=$X/cache/heal_inputs
L=/content/lang; PORT=8000
mkdir -p $L
[ -f $L/ask_on.jsonl ] || python -u "$X/heal/lang_data.py" prompts --ultrachat /content/uc/train/data-00000-of-00019.arrow \
    --out $L || { echo "prompts failed"; exit 1; }
if [ ! -f $L/answers.done ]; then
  pip install -q -U vllm 2>&1 | tail -1
  vllm serve Qwen/Qwen3.5-35B-A3B --served-model-name orig --port $PORT --max-model-len 12288 \
      --gpu-memory-utilization 0.88 --max-num-seqs 64 > $L/vllm.log 2>&1 &
  pid=$!
  for i in $(seq 1 480); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 $pid 2>/dev/null || { echo "vLLM exited:"; tail -30 $L/vllm.log; exit 1; }
    sleep 5
  done
  echo "== serving ($(date '+%H:%M'))"
  cd "$X/fix_task"
  python -u eval_chat.py --base-url "http://127.0.0.1:$PORT/v1" --model-name orig --no-seed --concurrency 64 --resume \
      --refs none --n-per-domain 100000 --tasks $L/ask_on.jsonl --out $L/ask_on.answers.jsonl --max-tokens 4000 > $L/on.txt 2>&1
  python -u eval_chat.py --base-url "http://127.0.0.1:$PORT/v1" --model-name orig --no-seed --concurrency 64 --resume \
      --refs none --n-per-domain 100000 --tasks $L/ask_off.jsonl --out $L/ask_off.answers.jsonl --max-tokens 1500 \
      --no-thinking > $L/off.txt 2>&1
  kill $pid; sleep 10; kill -9 $pid 2>/dev/null
  [ -s $L/ask_on.answers.jsonl ] && [ -s $L/ask_off.answers.jsonl ] || { echo "no answers:"; tail -5 $L/on.txt $L/off.txt; exit 1; }
  touch $L/answers.done
  echo "== answered ($(date '+%H:%M'))"
fi
python "$X/heal/lang_data.py" traces --out $L || { echo "traces failed"; exit 1; }
pip install -q 'transformers==5.17.0' 2>&1 | tail -1
cd "$X/moe" && cp ../heal/heal_train.py .
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u heal_train.py --student /content/student/g128b28160 \
  --ultrachat /content/uc/train/data-00000-of-00019.arrow --run "$IN" --out /content/wikil \
  --traces "$X/cache/honesty/traces_mix2.jsonl" $L/traces_lang.jsonl --mix-traces --init-lora "$IN/lora_6e6.pt" \
  --g-share 0.3 --max-start-kl 0.3 --text /content/wiki_text.jsonl --text-share 0.4 --wiki-run /content/wikirun \
  --milestones 2e6,4e6,1e9 --max-hours 1.8 2>&1 | grep --line-buffered -v 'falling back'
ls -la /content/wikil
echo "W_DONE $(date '+%H:%M')"
