# SolarFlareForcast status
Updated: 2026-10-30 00:15, DESKTOP-BBASC81

## Headline
R4 done. Resetting the proton output head before fine-tuning keeps the X-ray gain and recovers ~40% of the small proton cost of synthetic pre-training. This model has the lowest validation error, so it is the new best: −13.8% X-ray error vs persistence (−22.8% during flares), −1.4% vs the real-only physics CNN.

## Best model
`real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr` (3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, pre-trained on synthetic telemetry, proton output head reset, fine-tuned on real GOES-16 NCEI L2 data 2022-04 → 2023-12.
Real test: X-ray RMSE 0.1530 ± 0.0006 dex (persistence 0.1776, plain CNN 0.1569 ± 0.0006); active 0.176; protons 0.1206 ± 0.0003 (persistence 0.159); quiet roughness 2.8e-4 dex/min²; TSS ≥M 0.349 ± 0.002.

## Latest results
Real GOES-16 test split (2024-07 → 2025-03), mean ± std over 3 seeds, RMSE in dex vs observations:

| Model | X-ray | X-ray active | Protons | Quiet roughness | TSS ≥M |
|---|---|---|---|---|---|
| Persistence | 0.1776 | 0.228 | 0.159 | 0 | 0.33 |
| CNN (MSE) | 0.1569 ± 0.0006 | 0.181 | 0.118 | 6.0e-4 | 0.336 ± 0.002 |
| CNN + physics | 0.1552 ± 0.0006 | 0.179 | 0.119 | 3.0e-4 | 0.341 ± 0.010 |
| + fine-tuned from synthetic (R3) | 0.1533 ± 0.0005 | 0.177 | 0.122 ± 0.001 | 2.3e-4 | 0.345 ± 0.006 |
| + proton head reset (R4) | 0.1530 ± 0.0006 | 0.176 | 0.1206 ± 0.0003 | 2.8e-4 | 0.349 ± 0.002 |

## Running now
Nothing. R4 ran 22:54–23:58 on 2026-10-29.

## Waiting on user
1. Next direction: probabilistic forecasts (quantile/ensemble, Brier skill) or GOES-18 data (more years, cross-satellite test)?

## Next up
Depends on the choice above. README, REPORT (R4), onboarding notebook and CV_BULLETS.md are updated with R4.
