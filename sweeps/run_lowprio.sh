#!/usr/bin/env bash
# Run each line of stdin as a train_real.jl job, one at a time, at BelowNormal Windows priority, then analyze_real.
# A watcher lowers only julia.exe processes descended from this script (matched by parent process; juliaup hides the
# script name in julia's command line), so other projects' Julia processes are never touched.
# sweeps/results/.lowprio_done is deleted at the start and created only after analyze_real exits (other sessions wait on it).
cd "$(dirname "$0")/.."
ROOT=$(cat /proc/$$/winpid)
rm -f sweeps/results/.lowprio_done
( while [ ! -f sweeps/results/.lowprio_done ]; do
    powershell -NoProfile -ExecutionPolicy Bypass -File sweeps/lower_descendants.ps1 -Root "$ROOT" >/dev/null 2>&1
    sleep 10; done ) &
while read -r a b c; do
  log="sweeps/results/log_train_real_${a}_${b}_${c}.txt"
  julia -t 1 --project=. sweeps/train_real.jl "$a" "$b" "$c" > "$log" 2>&1 && echo "DONE $a $b $c" || echo "FAILED $a $b $c"
done
julia -t 1 --project=. sweeps/analyze_real.jl > sweeps/results/analyze_real_final.txt 2>&1; echo "analyze_real exit $?"
touch sweeps/results/.lowprio_done
date +%H:%M
