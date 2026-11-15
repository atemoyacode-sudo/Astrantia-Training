# xmodel rehearsal 6h: cut an uneven-width student (experts of different widths, stored unpadded; moe/uneven.py) from
# the original Qwen3.5-35B-A3B (read over HTTP) using the chat statistics saved on the Mac (stats.npz), no GPU.
# The checkpoint (bf16, ~19 GB for 9.4B parameters) is this kernel's output; heal_train.py loads it.
import glob, os, shutil, subprocess, sys
SHAPE = "g128b28160"
W, inp = "/kaggle/working", os.path.dirname(glob.glob("/kaggle/input/**/build_shape.py", recursive=True)[0])
code = "/tmp/code"
os.makedirs(f"{code}/moe", exist_ok=True)
for f in ("common.py", "merge_pass.py", "width_pass.py", "build_shape.py", "uneven.py"):
    shutil.copy(f"{inp}/{f}", f"{code}/moe/{f}")
shutil.copy(f"{inp}/remote_st.py", f"{code}/remote_st.py")
def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)
sh("df -h /kaggle/working /tmp; nproc; free -g")
sh(f"{sys.executable} -m pip install -q 'transformers==5.17.0' 2>&1 | tail -2")
sh(f"cd {code}/moe && {sys.executable} -u build_shape.py --source Qwen/Qwen3.5-35B-A3B --run {inp} --shapes {SHAPE} "
   f"--out {W} --from-stats {inp}/stats.npz --device cpu 2>&1 | grep --line-buffered -v 'falling back'")
sh(f"ls -la {W}/{SHAPE} | head; du -sh {W}/{SHAPE}; cat {W}/{SHAPE}/config.json | tail -8")
