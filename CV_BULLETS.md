# CV bullets

Numbers are from the real GOES-16 test split (Jul 2024 – Mar 2025, sealed; mean of 3 seeds), see `sweeps/REPORT.md` R1–R4 and P1.
Updated 2026-10-30 (best model: R4, proton-head reset; probabilistic: P1).

**Physics-Constrained Solar Flare & SEP Forecasting** (independent project; Julia, Flux.jl, CairoMakie)

- Built a residual 1D-CNN in Julia/Flux that forecasts GOES soft X-ray and ≥10 MeV proton flux 60 min ahead at 1-min
  cadence. On 9 months of held-out GOES-16 data, including the Oct 2024 X9.0 flare, it cuts X-ray forecast error by
  14% vs the operational persistence baseline (23% during flares) and proton error by 24%. A learned quantile head adds
  probabilistic forecasts that beat static error bars, with 12% narrower 90% intervals during flares at equal coverage.
- Designed a physics-informed loss (one-sided plasma-cooling bound, regime-gated quiet-Sun curvature, decay-phase
  convexity), validated by ablation on Hawkes-process synthetic telemetry. Pre-training on the simulator and
  fine-tuning on real data beat real-only training by 1.4% (>3σ across seeds), with quiet-period forecasts 53% smoother
  than an unconstrained CNN.
- Built a validated pipeline from NOAA NCEI GOES-R netCDF archives: per-minute quality-flag filtering, reconstruction
  of the ≥10 MeV integral proton flux from 13 differential SGPS channels (median ratio 1.19 to NOAA's 2024 SEP event
  list), leakage-free chronological splits with guard gaps, and a reproducible Pkg environment with unit tests.
