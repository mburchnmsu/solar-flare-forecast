#!/usr/bin/env bash
# Run each line of stdin as a train_real.jl job, one at a time, at BelowNormal Windows priority, then analyze_real.
# A watcher lowers the priority of every julia.exe started after this script began (other projects' are older).
cd "$(dirname "$0")/.."
T0=$(powershell -NoProfile -Command "(Get-Date).ToString('o')")
( while [ ! -f sweeps/results/.lowprio_done ]; do
    powershell -NoProfile -Command "Get-Process julia -ErrorAction SilentlyContinue | Where-Object { \$_.StartTime -ge [datetime]'$T0' -and \$_.PriorityClass -ne 'BelowNormal' } | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" 2>/dev/null
    sleep 10; done ) &
rm -f sweeps/results/.lowprio_done
while read -r a b c; do
  log="sweeps/results/log_train_real_${a}_${b}_${c}.txt"
  julia -t 1 --project=. sweeps/train_real.jl "$a" "$b" "$c" > "$log" 2>&1 && echo "DONE $a $b $c" || echo "FAILED $a $b $c"
done
julia -t 1 --project=. sweeps/analyze_real.jl > sweeps/results/analyze_real_final.txt 2>&1; echo "analyze_real exit $?"
touch sweeps/results/.lowprio_done
date +%H:%M
