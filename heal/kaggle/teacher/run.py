# xmodel light healing, step 1: the teacher (original Qwen3.5-35B-A3B, Q4_K_M GGUF, llama.cpp on 2x T4) answers the
# questions in prompts.jsonl with thinking on. Every answer is appended to /kaggle/working/traces.jsonl as it arrives.
# Stops at MAX_HOURS or once the finished answers (thinking closed, not cut off) hold TARGET_TOKENS completion tokens.
import glob, json, os, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = "/kaggle/working"
MAX_HOURS, TARGET = float(os.environ.get("MAX_HOURS", "7")), int(float(os.environ.get("TARGET_TOKENS", "9e6")))
MAX_TOKENS, NP = 8192, 16
T0 = time.time()


def log(m):
    s = time.strftime("%H:%M:%S ") + str(m)
    print(s, flush=True)
    open(f"{OUT}/teacher.log", "a").write(s + "\n")


def sh(c, timeout=3600):
    t = time.time()
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f"$ {c[:160]} -> {r.returncode} ({time.time() - t:.0f}s)\n{(r.stdout + r.stderr)[-800:]}")
    return r.returncode


binsrc = os.path.dirname(glob.glob("/kaggle/input/**/llama-server", recursive=True)[0])
BIN = "/tmp/bin"
sh(f"mkdir -p {BIN} && cp {binsrc}/* {BIN}/ && chmod +x {BIN}/llama-server && cd {BIN} && "
   "for f in lib*.so.*.*; do b=${f%%.so.*}; v=${f#*.so.}; ln -sf $f $b.so.${v%%.*}; ln -sf $f $b.so; done")
# the prebuilt llama.cpp links CUDA 12 (libcudart.so.12, libcublas.so.12); newer Kaggle images ship another CUDA
sh("pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 2>&1 | tail -1")
import site
nv = [d for sp in site.getsitepackages() for d in glob.glob(f"{sp}/nvidia/*/lib")]
os.environ["LD_LIBRARY_PATH"] = ":".join([BIN] + nv + [os.environ.get("LD_LIBRARY_PATH", "")])
sh(f"ldd {BIN}/llama-server | grep -i 'not found\\|cuda\\|cublas'")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp")
sh("pip install -q -U 'huggingface_hub[cli]' 2>&1 | tail -1; hf download unsloth/Qwen3.5-35B-A3B-GGUF "
   "Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir /tmp/q", timeout=7200)

srv = None
for np_ in (NP, 8):
    lf = open(f"{OUT}/server.log", "a")
    srv = subprocess.Popen(f"{BIN}/llama-server -m /tmp/q/Qwen3.5-35B-A3B-Q4_K_M.gguf --port 8100 --jinja -ngl 99 -sm layer "
                           f"-c {np_ * (MAX_TOKENS + 1024)} -np {np_} -fa on --reasoning-format none -ub 512 -b 2048",
                           shell=True, stdout=lf, stderr=subprocess.STDOUT)
    up = False
    for _ in range(120):
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=5)
            up = True
            break
        except Exception:
            time.sleep(10)
    if up:
        NP = np_
        break
    log(f"server did not start with -np {np_}")
    srv.kill()
    sh(f"tail -30 {OUT}/server.log")
if not up:
    raise SystemExit("llama-server did not start; see server.log")
log(f"server up with {NP} slots ({time.time() - T0:.0f}s)")
sh("nvidia-smi --query-gpu=memory.used --format=csv")

prompts = [json.loads(l) for l in open(glob.glob("/kaggle/input/**/prompts.jsonl", recursive=True)[0])]
done_path = f"{OUT}/traces.jsonl"
done = {json.loads(l)["id"] for l in open(done_path)} if os.path.exists(done_path) else set()
todo = [p for p in prompts if p["id"] not in done]
lock, stop = threading.Lock(), threading.Event()
stats = {"n": 0, "kept": 0, "kept_tokens": 0, "all_tokens": 0}


def one(p):
    if stop.is_set():
        return
    body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": MAX_TOKENS, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    t = time.time()
    try:
        req = urllib.request.Request("http://127.0.0.1:8100/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        ch, use = r["choices"][0], r.get("usage", {})
        text = ch["message"].get("content") or ""
        row = {**p, "completion": text, "finish_reason": ch.get("finish_reason"),
               "completion_tokens": use.get("completion_tokens"), "prompt_tokens": use.get("prompt_tokens"),
               "seconds": round(time.time() - t, 1)}
    except Exception as e:
        row = {**p, "error": str(e)[:300]}
    with lock:
        open(done_path, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["n"] += 1
        ct = row.get("completion_tokens") or 0
        stats["all_tokens"] += ct
        if row.get("finish_reason") == "stop" and "</think>" in row.get("completion", ""):
            stats["kept"] += 1
            stats["kept_tokens"] += ct
        if stats["n"] % 25 == 0:
            el = time.time() - T0
            log(f"{stats['n']} answered, {stats['kept']} kept, kept tokens {stats['kept_tokens']:,}, "
                f"all tokens {stats['all_tokens']:,} ({el / 3600:.2f} h)")
        if stats["kept_tokens"] >= TARGET or time.time() - T0 > MAX_HOURS * 3600:
            stop.set()


tg = time.time()
with ThreadPoolExecutor(NP) as ex:
    futs = [ex.submit(one, p) for p in todo]
    for f in as_completed(futs):
        pass
el = time.time() - tg
log(f"finished: {stats} in {el / 3600:.2f} h, {stats['all_tokens'] / max(el, 1):.0f} completion tokens/s overall")
srv.kill()
