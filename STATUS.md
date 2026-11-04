# SolarFlareForcast status
Updated: 2026-11-03 22:45, DESKTOP-BBASC81 (ThinkPad)

## Headline
C1 done: isotonic recalibration (fit on validation) improves every event-probability score; the learned head stays best (Brier skill ≥M1 0.350, ≥S1 0.958). Tonight: GOES-18 experiment G1 (download running, then 3 training seeds).

## Best model
Point: `real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr` (R4, 3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, pre-trained on synthetic telemetry, proton output head reset, fine-tuned on real GOES-16 NCEI L2 data 2022-04 → 2023-12. Real test: X-ray RMSE 0.1530 ± 0.0006 dex (persistence 0.1776); active 0.176; protons 0.1206 ± 0.0003 (persistence 0.159); TSS ≥M 0.349 ± 0.002.
Probabilistic: the same model + learned 5/25/50/75/95% quantile head + isotonic recalibration of event probabilities. X-ray pinball 0.0279 ± 0.0001 dex; 90% interval coverage 0.925; Brier skill ≥M1 0.350, new M crossings 0.089, ≥S1 0.958.

## Latest results
C1, real GOES-16 test split (2024-07 → 2025-03), Brier skill raw → recalibrated, mean over 3 seeds:

| Model | ≥M1 | ≥M1, new crossings | ≥S1 |
|---|---|---|---|
| Persistence + static bars | 0.311 → 0.328 | 0.040 → 0.064 | 0.917 → 0.951 |
| Best model + static bars | 0.317 → 0.344 | 0.046 → 0.084 | 0.922 → 0.957 |
| Best model + learned head | 0.325 → 0.350 | 0.056 → 0.089 | 0.930 → 0.958 |

≥S1 is now calibrated; ≥M1 is still low on average (0.098 vs 0.129 observed) because of missed onsets.

## Running now
G1 (approved by the user via the hub, 2026-11-03): GOES-18 NCEI download 2022-09 → 2026-10 started 22:27 (network-bound, done ~23:50), then 3 seeds of the R4 recipe trained on GOES-16 + GOES-18, one at a time at low priority (~3.5–4 GB, 1 thread), then scoring on the GOES-16 test, the GOES-18 test (same period) and a new GOES-18 period 2025-04 → 2026-10. Expected finish ~01:30–02:00.

## Waiting on user
Nothing.

## Next up
G1 results; then the write-up (onboarding notebook + CV bullets with C1 and G1).
