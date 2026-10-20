# SolarFlareForcast status
Updated: 2026-10-20 17:45, DESKTOP-BBASC81

## Headline
Real-data result confirmed over 3 seeds: on GOES-16 (test Jul 2024 – Mar 2025) the physics-loss CNN beats persistence by 12.6% on X-ray flux (21% during flares) and the plain CNN by 1.1% (≈3σ), and halves quiet-time roughness.

## Best model
`real_physp_1e4_0.01` (3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, trained on real GOES-16 NCEI L2 data 2022-04 → 2023-12.
Real test: X-ray RMSE 0.1552 ± 0.0006 dex (persistence 0.1776, plain CNN 0.1569 ± 0.0006); active 0.179; protons 0.119 (persistence 0.159).
(A single fine-tuned-from-synthetic run reaches 0.1537; unconfirmed.)

## Latest results
Real GOES-16 test split (2024-07 → 2025-03), mean ± std over 3 seeds, RMSE in dex vs observations:

| Model | X-ray | X-ray active | Protons | Quiet roughness | TSS ≥M |
|---|---|---|---|---|---|
| Persistence | 0.1776 | 0.228 | 0.159 | 0 | 0.33 |
| CNN (MSE) | 0.1569 ± 0.0006 | 0.181 | 0.118 | 6.0e-4 | 0.336 ± 0.002 |
| CNN + physics | 0.1552 ± 0.0006 | 0.179 | 0.119 | 3.0e-4 | 0.341 ± 0.010 |
| Fine-tuned from synthetic (n=1) | 0.1537 | 0.177 | 0.122 | 1.9e-4 | 0.350 |

## Running now
Nothing. All runs finished 17:44; the PC is free for Zodi.

## Waiting on user
- Optional: seeds 2–3 of the fine-tuned model to confirm the pre-training gain (2 runs, 1 worker, ~25 min each).

## Next up
Update README, onboarding notebook and CV bullets with the real-data numbers; then probabilistic forecasts or GOES-18 data (see SolOnboarding.ipynb §7).
