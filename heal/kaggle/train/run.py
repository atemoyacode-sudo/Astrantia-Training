# xmodel light healing, step 2: LoRA on the 96x256 student (float16, split over 2x T4) with the teacher's answers.
# MODE "smoke": speed / memory check on the UltraChat part only (no teacher answers needed), a few steps.
# MODE "train": the real run (teacher answers from the xmodel-heal-teacher kernel; RESUME from a state.pt if given).
import glob, os, shutil, subprocess, sys
MODE, RESUME = "train", "state.pt"
# teacher-answer kernels, in the order their answers are appended (keep the order when adding more and resuming)
TRACE_KERNELS = ["xmodel-heal-teacher"]
MILESTONES = "3e5,1e6,3e6,5e6,6e6"  # session 3: resumes at the end of the data (~4M), loop-exit phase to 6M
G_SHARE = 0.5  # the teacher made fewer answers than planned (about 1.8M tokens), so UltraChat fills half
W, PY = "/kaggle/working", sys.executable
inp = os.path.dirname(glob.glob("/kaggle/input/**/heal_train.py", recursive=True)[0])
student_in = os.path.dirname(glob.glob("/kaggle/input/**/96x256/config.json", recursive=True)[0])
code = "/tmp/code"
os.makedirs(f"{code}/moe", exist_ok=True)
for f in ("common.py", "lora_train.py", "heal_train.py"):
    shutil.copy(f"{inp}/{f}", f"{code}/moe/{f}")
shutil.copy(f"{inp}/remote_st.py", f"{code}/remote_st.py")
def sh(c, check=True):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=check).returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp; free -g")
# reading the 16 GB student straight from /kaggle/input took 26 min; copy it with 16 readers first
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
traces = [glob.glob(f"/kaggle/input/**/{k}/traces.jsonl", recursive=True)[0] for k in TRACE_KERNELS] if MODE == "train" else []
print("traces:", traces, flush=True)
args = (f"--student {student} --ultrachat /tmp/uc/train/data-00000-of-00019.arrow --run {inp} --out {W}/heal "
        f"--traces {' '.join(traces) if traces else '/dev/null'}")
args += f" --milestones {MILESTONES} --max-hours 11 --g-share {G_SHARE} --loop-exit"
if MODE == "smoke":
    args += " --smoke-steps 6 --step-tokens 16384"
elif RESUME:
    args += f" --resume {glob.glob(f'/kaggle/input/**/{RESUME}', recursive=True)[0]}"
sh(f"cd {code}/moe && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} -u heal_train.py {args} 2>&1 "
   f"| grep --line-buffered -v 'falling back'")
