# SolarFlareForcast status
Updated: 2026-10-30 22:10, DESKTOP-BBASC81

## Headline
P1 done: probabilistic forecasts work. A learned quantile head on the best model beats static error bars on every seed (X-ray pinball −2.7%, −15% vs persistence), and during flares its 90% intervals are 12% narrower at the same coverage. Event probabilities are biased low, so recalibration is next.

## Best model
Point: `real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr` (3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, pre-trained on synthetic telemetry, proton output head reset, fine-tuned on real GOES-16 NCEI L2 data 2022-04 → 2023-12. Real test: X-ray RMSE 0.1530 ± 0.0006 dex (persistence 0.1776); active 0.176; protons 0.1206 ± 0.0003 (persistence 0.159); TSS ≥M 0.349 ± 0.002.
Probabilistic: the same model + learned 5/25/50/75/95% quantile head. X-ray pinball 0.0279 ± 0.0001 dex; 90% interval coverage 0.925; Brier skill ≥M1 0.325, ≥S1 0.930.

## Latest results
P1, real GOES-16 test split (2024-07 → 2025-03), mean over 3 seeds, dex:

| Model | Pinball X | Pinball X, active | 90% cover X | 90% width X, active | BSS ≥M1 | BSS ≥S1 |
|---|---|---|---|---|---|---|
| Persistence + static bars | 0.0330 | 0.0465 | 0.911 | 0.676 | 0.311 | 0.917 |
| Best model + static bars | 0.0287 ± 0.0004 | 0.0353 | 0.915 | 0.526 | 0.317 | 0.922 |
| Best model + learned head | 0.0279 ± 0.0001 | 0.0339 | 0.925 | 0.461 | 0.325 | 0.930 |

Point-forecast table (R1–R4) is in README.md and sweeps/REPORT.md.

## Running now
Nothing. P1 ran 21:43–21:57. Claude Code's low-memory safeguard reaped its wrapper shell, but the Julia job finished normally.

## Waiting on user
1. Next: recalibrate event probabilities (cheap, analysis only), GOES-18 data, or wrap up the write-up?

## Next up
Isotonic recalibration of the ≥M1 / ≥S1 probabilities on validation (no training needed). Onboarding notebook and CV bullets are updated with P1.
