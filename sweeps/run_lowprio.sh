#!/usr/bin/env bash
# Run each line of stdin as a job, one at a time, at BelowNormal Windows priority, then an analysis script.
#   JOB      script run once per stdin line, with the line's words as arguments (default sweeps/train_real.jl)
#   ANALYZE  script run after all jobs (default sweeps/analyze_real.jl; "none" to skip)
# A watcher lowers only julia.exe processes descended from this script (matched by parent process; juliaup hides the
# script name in julia's command line), so other projects' Julia processes are never touched.
# sweeps/results/.lowprio_done is deleted at the start and created only after the last step exits (other sessions wait on it).
cd "$(dirname "$0")/.."
JOB=${JOB:-sweeps/train_real.jl}
ANALYZE=${ANALYZE:-sweeps/analyze_real.jl}
ROOT=$(cat /proc/$$/winpid)
rm -f sweeps/results/.lowprio_done
( while [ ! -f sweeps/results/.lowprio_done ]; do
    powershell -NoProfile -ExecutionPolicy Bypass -File sweeps/lower_descendants.ps1 -Root "$ROOT" >/dev/null 2>&1
    sleep 10; done ) &
while read -r -a args; do
  [ ${#args[@]} -eq 0 ] && continue
  log="sweeps/results/log_$(basename "$JOB" .jl)_$(IFS=_; echo "${args[*]}").txt"
  julia -t 1 --project=. "$JOB" "${args[@]}" > "$log" 2>&1 && echo "DONE ${args[*]}" || echo "FAILED ${args[*]}"
done
if [ "$ANALYZE" != none ]; then
  log="sweeps/results/$(basename "$ANALYZE" .jl)_final.txt"
  julia -t 1 --project=. "$ANALYZE" > "$log" 2>&1; echo "$(basename "$ANALYZE") exit $?"
fi
touch sweeps/results/.lowprio_done
date +%H:%M
