#!/bin/sh
# Run long jobs on a Colab session in the background and check on them with short calls.
# `colab exec` waits for the cell to finish and can hang or time out, so each call here returns at once.
#
#   xmodel/colab_bg.sh start  <session> <name> '<shell command>'   # nohup on the VM, log in /content/logs/<name>.log
#   xmodel/colab_bg.sh tail   <session> <name> [lines]             # last lines of the log, and whether it is still running
#                                                                  # (also keeps the session alive: poll at least every ~10 minutes)
#   xmodel/colab_bg.sh sh     <session> '<short shell command>'    # run a quick command and print its output
set -eu
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python3 -c 'import certifi; print(certifi.where())')}"
cmd="$1"; session="$2"; shift 2
tmp=$(mktemp -t colab_bg).py
trap 'rm -f "$tmp"' EXIT

case "$cmd" in
  start)
    name="$1"; job="$2"
    cat > "$tmp" <<EOF
import subprocess, os
os.makedirs("/content/logs", exist_ok=True)
job = r'''$job'''
p = subprocess.Popen(["bash", "-lc", f"({job}) > /content/logs/$name.log 2>&1; echo EXIT=\$? >> /content/logs/$name.log"],
                     start_new_session=True)
open("/content/logs/$name.pid", "w").write(str(p.pid))
print("started $name pid", p.pid)
EOF
    ;;
  tail)
    # Read the log through the file API (`colab download`), not the kernel: the kernel API often answers 503 while a
    # job keeps the VM busy, but downloads still work.
    name="$1"; lines="${2:-20}"
    # Also touch the kernel: a session whose kernel goes ~20 minutes without an execution is reclaimed, even while a
    # background job is running (2026-11-08: lost a LoRA run this way after polling only through downloads).
    echo 'print("alive")' > "$tmp"
    timeout 90 colab exec -s "$session" --timeout 60 -f "$tmp" >/dev/null 2>&1 || echo "[heartbeat failed]"
    log=$(mktemp -t colab_log)
    if colab download -s "$session" "/content/logs/$name.log" "$log" >/dev/null 2>&1; then
      tail -n "$lines" "$log" | grep -v -e 'falling back' -e 'Loading weights'
      echo
      if tail -n 3 "$log" | grep -q 'EXIT='; then echo "[finished]"; else echo "[running]"; fi
    else
      echo "[unreachable] could not download /content/logs/$name.log"
    fi
    rm -f "$log"
    exit 0
    ;;
  sh)
    job="$1"
    cat > "$tmp" <<EOF
import subprocess
r = subprocess.run(["bash", "-lc", r'''$job'''], capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
EOF
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
# `colab exec` can hang when the connection to the kernel is lost (2026-11-13: ten minutes): give up after 150 s
timeout 150 colab exec -s "$session" --timeout 120 -f "$tmp" 2>&1 | grep -v -e 'new version of Colab CLI' -e "colab update" -e 'enable_update_check'
