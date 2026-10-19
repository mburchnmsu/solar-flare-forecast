#!/usr/bin/env bash
# Run jobs (one per line on stdin: the script's arguments) with N parallel single-thread Julia workers.
# Before each launch, waits until at least MIN_FREE_GB of RAM is free (the Zodi project shares this PC).
#     printf "mse 1\nphys_1e4 1\n" | bash sweeps/pool.sh 2                         # synthetic sweeps
#     printf "phys_1e4 1 phys_1e4_s1\n" | SCRIPT=sweeps/train_real.jl bash sweeps/pool.sh 2
cd "$(dirname "$0")/.."
N=${1:-2}
export MIN_FREE_GB=${MIN_FREE_GB:-4}
export SCRIPT=${SCRIPT:-sweeps/train_variant.jl}
export PREFIX=$(basename "$SCRIPT" .jl)
xargs -P "$N" -L 1 bash -c '
  while [ "$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")" -lt "$MIN_FREE_GB" ]; do sleep 120; done
  name=$(echo "$*" | tr " " "_")
  [ "$PREFIX" = train_variant ] && log="sweeps/results/log_$1_s$2.txt" || log="sweeps/results/log_${PREFIX}_${name}.txt"
  julia -t 1 --project=. "$SCRIPT" "$@" > "$log" 2>&1 && echo "DONE $*" || echo "FAILED $*"' _
