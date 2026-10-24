# SolarFlareForcast status
Updated: 2026-10-24 01:45, DESKTOP-BBASC81

## Headline
Synthetic pre-training confirmed over 3 seeds: the physics CNN pre-trained on synthetic telemetry and fine-tuned on real GOES-16 data is the new best model, beating persistence by 13.7% on X-ray flux (22.5% during flares) and the real-only physics CNN by 1.2% (≈3σ).

## Best model
`real_physp_1e4_0.01_ft-physp_1e4_0.01_s1` (3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, pre-trained on synthetic telemetry, then fine-tuned on real GOES-16 NCEI L2 data 2022-04 → 2023-12.
Real test: X-ray RMSE 0.1533 ± 0.0005 dex (persistence 0.1776, plain CNN 0.1569 ± 0.0006); active 0.177; protons 0.122 (persistence 0.159); quiet roughness 2.3e-4 dex/min².

## Latest results
Real GOES-16 test split (2024-07 → 2025-03), mean ± std over 3 seeds, RMSE in dex vs observations:

| Model | X-ray | X-ray active | Protons | Quiet roughness | TSS ≥M |
|---|---|---|---|---|---|
| Persistence | 0.1776 | 0.228 | 0.159 | 0 | 0.33 |
| CNN (MSE) | 0.1569 ± 0.0006 | 0.181 | 0.118 | 6.0e-4 | 0.336 ± 0.002 |
| CNN + physics | 0.1552 ± 0.0006 | 0.179 | 0.119 | 3.0e-4 | 0.341 ± 0.010 |
| CNN + physics, fine-tuned from synthetic | 0.1533 ± 0.0005 | 0.177 | 0.122 ± 0.001 | 2.3e-4 | 0.345 ± 0.006 |

## Running now
Nothing (R3 fine-tune seeds 2–3 finished 01:37).

## Waiting on user
Nothing.

## Next up
Update the onboarding notebook and CV bullets with the fine-tuned numbers. Possible follow-up: keep the X-ray gain without the small proton cost (proton-specific fine-tune); then probabilistic forecasts or GOES-18 data (see SolOnboarding.ipynb §7).
