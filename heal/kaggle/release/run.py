# xmodel release (README 14): the chosen LoRA folded into the uneven cut g128b28160, checked, and uploaded to two
# PRIVATE Hugging Face repos (making them public is the owner's action):
#   <user>/Sharpen-Qwen-3.5-9B-A2.6B-Preview         the uneven checkpoint (bf16, ~19 GB) + uneven.py
#   <user>/Sharpen-Qwen-3.5-9B-A2.6B-Preview-padded  pad_uneven.py's plain 128-expert checkpoint (bf16, ~35 GB)
# The HF token is a Kaggle secret, given only to runs started from the editor ("Save Version"); a run pushed by the API
# stops at the first step. 2x T4 for the KL check and the padding's router statistics.
import glob, json, os, shutil, subprocess, sys, time
MODE = "gguf"  # "release" (fold + upload, done 11-15 20:29) or "gguf" (Q4_K_M of the padded repo)
CHOICE = "wikila"  # README 14, decided 11-15 18:28 (wikil-a: lowest wrong share 0.68; MMLU-Pro 123 vs run 3 111)
LORAS = {"run3": ("lora_run3.pt", 0.1371870917888979), "wikila": ("lora_wikila.pt", 0.14263378442265093),
         "6e6": ("lora_6e6.pt", None)}  # (file, the run's own kl_AB at the end; None = not recorded)
REPO = "Sharpen-Qwen-3.5-9B-A2.6B-Preview"
W, PY = "/kaggle/working", sys.executable


def sh(c, check=True):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=check).returncode


# 1. the token first: without it nothing below is worth running
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print(f"HF_TOKEN not available ({type(e).__name__}): start this run from the editor with Save Version", flush=True)
    raise SystemExit(0)
# pinned: the newest huggingface_hub on Kaggle (11-15) has no upload_large_folder; 1.33 is the Mac's, which has it
sh(f"{PY} -m pip install -q 'huggingface_hub==1.33.0' hf_xet 2>&1 | tail -1")
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)


def upload(repo, folder):  # resumable large upload when this hub has it, else the plain one
    api.create_repo(repo, private=True, exist_ok=True)
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model", private=True)
    else:
        api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model")


user = api.whoami()["name"]
if CHOICE not in LORAS:
    print("no model chosen yet (CHOICE is None): nothing to do", flush=True)
    raise SystemExit(0)
LORA, EXPECT_KL = LORAS[CHOICE]
print(f"choice {CHOICE} ({LORA}); token works", flush=True)

if MODE == "gguf":
    # README 14 (11-15 20:40): llama.cpp's converter on the padded checkpoint (a plain
    # Qwen3.5-MoE), bf16 GGUF, then Q4_K_M; a short CPU check that it answers; the Q4_K_M file goes into the main repo.
    from huggingface_hub import snapshot_download
    t0 = time.time()
    src = snapshot_download(f"{user}/{REPO}-padded", local_dir="/tmp/padded", token=TOKEN, max_workers=8)
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)
    sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && cd /tmp/llama.cpp && git log -1 --format='%h %cd'")
    # not the converter's requirements file: it pins transformers 4.x, which refuses huggingface_hub 1.x (11-15 21:10)
    # and may not know Qwen3.5; the gguf package from the clone and transformers 5 instead
    sh(f"{PY} -m pip install -q /tmp/llama.cpp/gguf-py sentencepiece protobuf 'transformers==5.20.2' 'huggingface_hub==1.33.0' 2>&1 | tail -1")
    os.makedirs("/tmp/gguf", exist_ok=True)
    bf = "/tmp/gguf/Sharpen-Qwen-3.5-9B-A2.6B-Preview-BF16.gguf"
    sh(f"cd /tmp/llama.cpp && {PY} convert_hf_to_gguf.py /tmp/padded --outtype bf16 --outfile {bf} 2>&1 | tail -25", check=False)
    if not os.path.exists(bf):  # (the exit code of a pipe into tail says nothing)
        raise SystemExit("the converter did not write the GGUF; nothing uploaded")
    sh("cd /tmp/llama.cpp && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /dev/null && "
       "cmake --build build --target llama-quantize llama-cli -j$(nproc) 2>&1 | tail -2")
    q4 = "/tmp/gguf/Sharpen-Qwen-3.5-9B-A2.6B-Preview-Q4_K_M.gguf"
    sh(f"/tmp/llama.cpp/build/bin/llama-quantize {bf} {q4} Q4_K_M 2>&1 | tail -4")
    if not os.path.exists(q4):
        raise SystemExit("llama-quantize did not write the Q4_K_M file; nothing uploaded")
    sh(f"ls -la /tmp/gguf; rm -rf /tmp/padded")
    # the check: one short question, thinking off, greedy, on the CPU
    sh(f"cd /tmp && timeout 900 /tmp/llama.cpp/build/bin/llama-cli -m {q4} -no-cnv -n 120 --temp 0 -t $(nproc) "
       f"-p '<|im_start|>user\nWhat is 17 times 23? Answer with the number and one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "
       f"2>/dev/null | tail -8", check=False)
    os.remove(bf)  # the bf16 GGUF is as large as the padded repo; only Q4_K_M goes up
    # into the main repo itself (20:50: downloadable from the same page)
    api.upload_file(path_or_fileobj=q4, path_in_repo=os.path.basename(q4), repo_id=f"{user}/{REPO}", repo_type="model")
    print(f"GGUF_DONE in {time.time() - t0:.0f}s", flush=True)
    raise SystemExit(0)

inp = os.path.dirname(glob.glob("/kaggle/input/**/fold_lora_uneven.py", recursive=True)[0])
src_student = os.path.dirname(glob.glob("/kaggle/input/**/g128b28160/config.json", recursive=True)[0])
student, padded, code = "/tmp/student", "/tmp/padded", "/tmp/code"
for dd in ("moe", "heal"):
    os.makedirs(f"{code}/{dd}", exist_ok=True)
for f in ("pad_uneven.py", "uneven.py", "lora_train.py", "common.py"):
    shutil.copy(f"{inp}/{f}", f"{code}/moe/{f}")
for f in ("heal_train.py", "fold_lora_uneven.py"):
    shutil.copy(f"{inp}/{f}", f"{code}/heal/{f}")
shutil.copy(f"{inp}/remote_st.py", f"{code}/remote_st.py")
os.makedirs("/tmp/run", exist_ok=True)
for f in ("data.npz", "teacher.npz"):
    shutil.copy(f"{inp}/{f}", f"/tmp/run/{f}")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp; free -g; nproc")
t0 = time.time()

# 2. copy the student to local disk (reading the experts from /kaggle/input stalled on 14 Nov) and fold the LoRA
sh(f"mkdir -p {student} && cd {src_student} && ls | xargs -P 16 -I{{}} cp {{}} {student}/ && du -sh {student}")
sh(f"{PY} -m pip install -q 'transformers==5.17.0' accelerate flash-linear-attention 2>&1 | tail -1")
sh(f"cd {code}/heal && {PY} -u fold_lora_uneven.py --student {student} --lora {inp}/{LORA}")
shutil.copy(f"{student}/fold_check.json", f"{W}/fold_check.json")

# 3. check: KL of the folded model to the original on the held-out windows must match the run's own number
check = f"""
import sys, json, numpy as np, torch
sys.path.insert(0, '{code}/heal'); sys.path.insert(0, '{code}/moe')
from heal_train import load_student, eval_kl
m = load_student('{student}', torch.cuda.device_count())
data, teacher = np.load('/tmp/run/data.npz'), np.load('/tmp/run/teacher.npz')
with torch.no_grad():
    kl = eval_kl(m, data, teacher, next(m.parameters()).device)
print('KL_AB', float(kl.mean()), len(kl))
json.dump({{'kl_AB': float(kl.mean()), 'n': len(kl)}}, open('{W}/kl_check.json', 'w'))
"""
open("/tmp/kl_check.py", "w").write(check)
sh(f"cd {code}/heal && {PY} -u /tmp/kl_check.py 2>&1 | grep --line-buffered -v -e 'falling back' -e 'Loading weights' -e MISSING")
kl = json.load(open(f"{W}/kl_check.json"))["kl_AB"]
print(f"folded KL_AB {kl:.4f}; the run's own {EXPECT_KL}", flush=True)
if EXPECT_KL is not None and abs(kl - EXPECT_KL) > 0.01:
    raise SystemExit("the folded model does not match the trained LoRA; nothing uploaded")

# 4. upload the uneven checkpoint (private) with the code that loads it
for f in ("uneven.py",):
    shutil.copy(f"{inp}/{f}", f"{student}/{f}")
upload(f"{user}/{REPO}", student)
print(f"uploaded the uneven checkpoint in {time.time() - t0:.0f}s", flush=True)

# 5. the padded checkpoint from the folded student (no --lora: it is already in the weights), bf16, then upload
sh(f"cd {code}/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} -u pad_uneven.py --student {student} "
   f"--run /tmp/run --out {padded} --save-dtype bfloat16 --cal-prompts {inp}/ask_things_on.jsonl "
   f"--max-cal-prompts 120 --check-prompts {inp}/chat_eval_set.jsonl --no-check 2>&1 "
   f"| grep --line-buffered -v -e 'falling back' -e 'Loading weights' -e MISSING")
shutil.copy(f"{padded}/pad_check.json", f"{W}/pad_check.json")
hits = json.load(open(f"{padded}/pad_check.json"))["dummy_hits"]
print(f"padding done; dummy hits {hits}", flush=True)
if hits:
    raise SystemExit("a dummy expert would be picked: the padded model is not the cut model; not uploaded")
upload(f"{user}/{REPO}-padded", padded)
print(f"RELEASE_DONE in {time.time() - t0:.0f}s", flush=True)
