# SolarFlareForcast status
Updated: 2026-10-19 12:38, DESKTOP-BBASC81

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
- Per-channel λ (proton terms × 0.1 / 0.01 / 0) + single-term ablations at λ = 1e4: 6 synthetic runs, 2 workers, started 12:20, expected ~14:30.
- Download of GOES-16 NCEI L2 data 2022-01 → 2025-03 (~2.8 GB, network only), expected ~15:00.

## Waiting on user
Nothing. Approved today: CLAUDE.md notebook path (done), the 2-worker ablation run, real-data training.

## Next up
Real-data training on GOES-16 (train 2022–23, val 2024 H1, test 2024-07 → 2025-03): MSE vs physics vs fine-tuned-from-synthetic, 2 workers.
Real-data loader validated: XRS-B matches SWPC real-time to 0.001 dex (median); reconstructed ≥10 MeV SEP peaks vs NOAA's 2024 event list: median ratio 1.19, scatter 0.22 dex (n = 8).
