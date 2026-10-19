# SolarFlareForcast status
Updated: 2026-10-18 22:40, DESKTOP-BBASC81

## Headline
Sweep 1 done (λ sweep + 3 seeds). On synthetic data the physics-loss CNN is the best X-ray forecaster (−14% RMSE vs persistence, −2% vs plain CNN), but zero-shot on real SWPC data it is 5% worse than persistence. Real-data validation is the next priority.

## Best model
`phys_1e4`: residual 1D-CNN (180 min in → 60 min out) + cooling-bound / quiet-curvature / convexity loss, λ = 1e4 (chosen on validation).
Synthetic test, 3 seeds: X-ray RMSE 0.2303 ± 0.0008 dex (persistence 0.268, plain CNN 0.2356 ± 0.0002); active periods 0.282 dex (persistence 0.377); quiet roughness −77% vs plain CNN; TSS ≥M 0.399.

## Latest results
Synthetic test split (last 13.5 of 90 days), mean ± std over 3 seeds, RMSE in dex vs noise-free truth:

| Model | X-ray | X-ray active | Protons | TSS ≥M |
|---|---|---|---|---|
| Persistence | 0.268 | 0.377 | 0.060 | 0.391 |
| CNN (MSE) | 0.2356 ± 0.0002 | 0.294 | 0.057 | 0.384 |
| CNN + naive smoothing (λ=30) | 0.2354 ± 0.0003 | 0.294 | 0.058 | 0.308 |
| CNN + physics loss (λ=1e4) | 0.2303 ± 0.0008 | 0.282 | 0.060 | 0.399 |

Zero-shot, real SWPC 7-day data (2026-10-12 → 10-19, max M1.4), X-ray RMSE: persistence 0.164, CNN 0.161, physics 0.173.

## Running now
Nothing.

## Waiting on user
- CLAUDE.md says dated notebooks live at the root; they're in notebooks/ now. Update it?
- Next compute (needs OK, Zodi has priority): per-channel λ + per-term ablation, ~8 runs, 2 workers × ~2 h.
- Direction: train on real NCEI GOES data next (adds NCDatasets.jl, a few GB of disk)?

## Next up
Real-data quality control (eclipse/dropout masking), per-channel λ, gate the convexity prior by flare/background ratio, then real-data training. Details: sweeps/REPORT.md, SolOnboarding.ipynb §6–7.
