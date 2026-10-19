#!/usr/bin/env bash
# Overnight chain approved 2026-10-18 (max 2 workers). The original pool already ran phys_1e5/phys_1e6, so this
# waits for all 8 phase-A runs (max 4 h) → select → seeds 2,3 of chosen → analyze → zero-shot.
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }
log "waiting for phase-A runs"
for i in $(seq 1 240); do
  n=0; for t in mse naive_0.3 naive_3 naive_30 phys_1e3 phys_1e4 phys_1e5 phys_1e6; do [ -f "sweeps/results/${t}_s1/run.jls" ] && n=$((n+1)); done
  [ $n -ge 8 ] && break; sleep 60
done
log "phase-A runs present: $n/8"
log "analyze (selection)"
julia -t 2 --project=. sweeps/analyze.jl > sweeps/results/analyze_A.txt 2>&1 || { log "ANALYZE FAILED"; exit 1; }
cat results/chosen.txt
log "phase B: seeds 2,3 of chosen tags"
awk '{print $2" 2\n"$2" 3"}' results/chosen.txt | bash sweeps/run_pool.sh 2
log "final analyze + zero-shot"
julia -t 2 --project=. sweeps/analyze.jl > sweeps/results/analyze_final.txt 2>&1 || { log "ANALYZE FAILED"; exit 1; }
julia -t 2 --project=. sweeps/zero_shot_swpc.jl > sweeps/results/zero_shot.txt 2>&1 || { log "ZERO-SHOT FAILED"; exit 1; }
log "CHAIN COMPLETE"
