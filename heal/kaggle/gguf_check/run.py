# Check that the published GGUF (Q4_K_M) answers sensibly: llama.cpp on the CPU, greedy, thinking off, 3 short prompts.
# The repo is public, so no token is needed.
import subprocess, os, sys
def sh(c, check=True):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=check).returncode
sh("nproc; free -g")
sh(f"{sys.executable} -m pip install -q 'huggingface_hub==1.33.0' hf_xet 2>&1 | tail -1")
from huggingface_hub import hf_hub_download
q4 = hf_hub_download(os.environ.get("HF_REPO", "Takenoko12345678/Sharpen-Qwen-3.5-9B-A2.6B-Preview"), "Sharpen-Qwen-3.5-9B-A2.6B-Preview-Q4_K_M.gguf", local_dir="/tmp/g")
sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && cd /tmp/llama.cpp && git log -1 --format='%h %cd'")
sh("cd /tmp/llama.cpp && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /dev/null && cmake --build build -j$(nproc) 2>&1 | tail -1")
sh("ls /tmp/llama.cpp/build/bin | head -60 | tr '\\n' ' '")
binp = "/tmp/llama.cpp/build/bin/llama-completion"
if not os.path.exists(binp):
    binp = "/tmp/llama.cpp/build/bin/llama-cli"
prompts = ["What is 17 times 23? Answer with the number and one short sentence.",
           "Write a Python function is_prime(n) that returns True if n is prime.",
           "Explain in two sentences why the sky is blue."]
for p in prompts:
    full = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    open("/tmp/p.txt", "w").write(full)
    sh(f"timeout 900 {binp} -m {q4} -f /tmp/p.txt -n 160 --temp 0 -t $(nproc) -no-cnv 2>/tmp/err.txt | tail -20; tail -3 /tmp/err.txt", check=False)
print("GGUF_CHECK_DONE", flush=True)
