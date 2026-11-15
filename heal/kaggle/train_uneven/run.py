# xmodel rehearsal 6h: LoRA on the uneven-width student g128b28160 (9.4B, float16, split over 2x T4), started from the
# 6M LoRA of 96x256 (--init-lora) with a new optimizer. Data: all teacher answers shuffled together (rehearsal 5 answers
# plus the right math top-up answers), UltraChat at 30% of the tokens. The week's GPU quota has under two hours left,
# so heal_train.py stops itself after MAX_HOURS and saves lora_final.pt with its KL.
import glob, os, shutil, subprocess, sys
SHAPE, MAX_HOURS, MILESTONES, G_SHARE = "g128b28160", 1.35, "5e5,1e6,1e9", 0.3
W, PY = "/kaggle/working", sys.executable
inp = os.path.dirname(glob.glob("/kaggle/input/**/heal_train.py", recursive=True)[0])
student_in = os.path.dirname(glob.glob(f"/kaggle/input/**/{SHAPE}/config.json", recursive=True)[0])
code = "/tmp/code"
os.makedirs(f"{code}/moe", exist_ok=True)
for f in ("common.py", "lora_train.py", "heal_train.py", "uneven.py"):
    shutil.copy(f"{inp}/{f}", f"{code}/moe/{f}")
shutil.copy(f"{inp}/remote_st.py", f"{code}/remote_st.py")
def sh(c, check=True):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=check).returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp; free -g")
# reading the student straight from /kaggle/input is slow; copy it with 16 readers first
student = "/tmp/student"
sh(f"mkdir -p {student} && cd {student_in} && date && ls | xargs -P 16 -I{{}} cp {{}} {student}/ && date && du -sh {student}")
sh(f"{PY} -m pip install -q 'transformers==5.17.0' accelerate flash-linear-attention 2>&1 | tail -2")
# fla's Triton kernels may not run on a T4; check once (forward + backward, float16) and fall back to torch if not
test = ("import torch; from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f; B,T,H,K=1,256,4,128; "
        "q,k,v=(torch.randn(B,T,H,K,device='cuda',dtype=torch.float16,requires_grad=True) for _ in range(3)); "
        "g=-torch.rand(B,T,H,device='cuda',dtype=torch.float32); b=torch.rand(B,T,H,device='cuda',dtype=torch.float16); "
        "o,_=f(q,k,v,g=g,beta=b,use_qk_l2norm_in_kernel=True); o.float().sum().backward(); print('fla ok', o.dtype)")
if sh(f"{PY} -c \"{test}\"", check=False) != 0:
    print("fla does not work here; using the torch path", flush=True)
    sh(f"{PY} -m pip uninstall -y flash-linear-attention fla-core 2>&1 | tail -1", check=False)
sh(f"hf download jiapingW/qwen3.5-35b-a3b-ultrachat-regen train/data-00000-of-00019.arrow --repo-type dataset --local-dir /tmp/uc")
traces = [glob.glob("/kaggle/input/**/xmodel-heal-teacher/traces.jsonl", recursive=True)[0], f"{inp}/traces_topup_right.jsonl"]
print("traces:", traces, flush=True)
args = (f"--student {student} --ultrachat /tmp/uc/train/data-00000-of-00019.arrow --run {inp} --out {W}/heal "
        f"--traces {' '.join(traces)} --mix-traces --init-lora {inp}/lora_6e6.pt --milestones {MILESTONES} "
        f"--max-hours {MAX_HOURS} --g-share {G_SHARE} --max-start-kl 0.3")
sh(f"cd {code}/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} -u heal_train.py {args} 2>&1 "
   f"| grep --line-buffered -v 'falling back'")
