#!/bin/bash
# Kill a model server before it can exhaust the Mac's memory (the caller restarts it and resumes).
#   bash heal/mem_guard.sh <server pid> [max GB of the server, default 18] [min free % system-wide, default 10] &
# The server's size is its physical footprint as `top` reports it (ps's RSS leaves out Metal memory).
PID=$1; MAX=${2:-18}; MINFREE=${3:-10}
while kill -0 "$PID" 2>/dev/null; do
  m=$(top -l 1 -pid "$PID" -stats mem 2>/dev/null | tail -1 | tr -d ' +-')
  case "$m" in *G) gb=${m%G};; *M) gb=$(( ${m%M} / 1024 ));; *) gb=0;; esac
  free=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
  if [ "${gb%.*}" -ge "$MAX" ] || { [ -n "$free" ] && [ "$free" -lt "$MINFREE" ]; }; then
    echo "== memory guard: server at ${m:-?}, system free ${free:-?}%: killing it ($(date '+%H:%M:%S'))"
    kill -9 "$PID" 2>/dev/null
    exit 0
  fi
  sleep 5
done
