#!/usr/bin/env bash
# Run each line of stdin as a train_real.jl job, one at a time, at BelowNormal Windows priority, then analyze_real.
# A watcher lowers the priority of this project's julia jobs only (matched on the script in their command line),
# so other projects' Julia processes are never touched.
cd "$(dirname "$0")/.."
( while [ ! -f sweeps/results/.lowprio_done ]; do
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='julia.exe'\" | Where-Object { \$_.CommandLine -match 'sweeps/(train|analyze)_real\.jl' } | ForEach-Object { \$p = Get-Process -Id \$_.ProcessId -ErrorAction SilentlyContinue; if (\$p -and \$p.PriorityClass -ne 'BelowNormal') { \$p.PriorityClass = 'BelowNormal' } }" 2>/dev/null
    sleep 10; done ) &
rm -f sweeps/results/.lowprio_done
while read -r a b c; do
  log="sweeps/results/log_train_real_${a}_${b}_${c}.txt"
  julia -t 1 --project=. sweeps/train_real.jl "$a" "$b" "$c" > "$log" 2>&1 && echo "DONE $a $b $c" || echo "FAILED $a $b $c"
done
julia -t 1 --project=. sweeps/analyze_real.jl > sweeps/results/analyze_real_final.txt 2>&1; echo "analyze_real exit $?"
touch sweeps/results/.lowprio_done
date +%H:%M
