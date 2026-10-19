#!/usr/bin/env bash
# Run "tag seed" jobs (one per line on stdin) with N parallel single-thread Julia workers.
# Before each launch, waits until at least MIN_FREE_GB of RAM is free (the Zodi project shares this PC).
#     printf "mse 1\nphys_1e4 1\n" | bash sweeps/run_pool.sh 2
cd "$(dirname "$0")/.."
N=${1:-2}
export MIN_FREE_GB=${MIN_FREE_GB:-4}
xargs -P "$N" -L 1 bash -c '
  while [ "$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")" -lt "$MIN_FREE_GB" ]; do sleep 120; done
  julia -t 1 --project=. sweeps/train_variant.jl "$0" "$1" > "sweeps/results/log_$0_s$1.txt" 2>&1 && echo "DONE $0 s$1" || echo "FAILED $0 s$1"'
