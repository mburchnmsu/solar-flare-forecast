# SolarFlareForcast status
Updated: 2026-10-18 21:31, DESKTOP-BBASC81

## Headline
Physics-constrained 1D-CNN forecasts the next 60 min of GOES X-ray and ≥10 MeV proton flux (synthetic data so far). The physics loss beats persistence and the plain CNN on X-rays, but in seed-1 results it collapses the proton forecast to persistence.

## Best model
`phys_1e4`: residual 1D-CNN (W=180 min → H=60 min) + cooling-bound/quiet-curvature/convexity loss, λ=1e4 (chosen on validation).
Test (seed 1): X-ray RMSE 0.231 dex (persistence 0.268, plain CNN 0.235); quiet-time roughness 3.1e-5 dex/min² (plain CNN 7.6e-4); M-class TSS 0.40.

## Latest results
Test split (last 13.5 of 90 synthetic days), seed 1, RMSE vs noise-free truth in dex:

| Model | X-ray | X-ray active | Protons | Quiet roughness | TSS ≥M |
|---|---|---|---|---|---|
| Persistence | 0.268 | 0.377 | 0.060 | 0 | 0.39 |
| CNN (MSE) | 0.235 | 0.292 | 0.054 | 7.6e-4 | 0.38 |
| CNN + naive smoothing (λ=30) | 0.235 | 0.291 | 0.057 | 6.0e-4 | 0.32 |
| CNN + physics loss (λ=1e4) | 0.231 | 0.285 | 0.060 | 3.1e-5 | 0.40 |

## Running now
Seeds 2–3 of the three chosen models (2 workers, started 21:31, expected done ~23:30), then final analysis + zero-shot test on live SWPC data.

## Waiting on user
- CLAUDE.md still says dated notebooks live at the root (they're in notebooks/ now). Update it?
- Proton penalty needs its own λ (follow-up run, 2 workers × ~1 h). Run it?

## Next up
Multi-seed summary → sweeps/REPORT.md, README, finish SolOnboarding.ipynb (results/kinks/future work), CV bullets; commit + push.
