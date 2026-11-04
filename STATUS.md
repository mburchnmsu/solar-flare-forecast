# SolarFlareForcast status
Updated: 2026-11-04 00:20, DESKTOP-BBASC81 (ThinkPad)

## Headline
G1 done: the best model (trained on GOES-16 only) keeps its skill on GOES-18 and on 19 months of new data (2025-04 → 2026-10): X-ray −13.4%, protons −20% vs persistence. Training on both satellites (76% more data) doesn't improve the X-ray forecast. C1 recalibration also done tonight.

## Best model
Point: `real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr` (R4, 3 seeds): residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, pre-trained on synthetic telemetry, proton output head reset, fine-tuned on real GOES-16 NCEI L2 data 2022-04 → 2023-12. GOES-16 test: X-ray RMSE 0.1530 ± 0.0006 dex (persistence 0.1776); protons 0.1206 (persistence 0.159); TSS ≥M 0.349.
Probabilistic: + learned quantile head + isotonic recalibration. X-ray pinball 0.0279 dex; Brier skill ≥M1 0.350, ≥S1 0.958.

## Latest results
G1, RMSE dex, mean over 3 seeds (R4 = GOES-16 only; G1 = trained on GOES-16 + GOES-18):

| Test set | X-ray: persistence / R4 / G1 | Protons: persistence / R4 / G1 |
|---|---|---|
| GOES-16, 2024-07 → 2025-03 | 0.1776 / 0.1530 / 0.1527 | 0.1594 / 0.1206 / 0.1208 |
| GOES-18, same period | 0.1761 / 0.1521 / 0.1516 | 0.1580 / 0.1190 / 0.1191 |
| GOES-18, new 2025-04 → 2026-10 | 0.1630 / 0.1411 / 0.1411 | 0.1647 / 0.1318 / 0.1266 |

GOES-18 vs GOES-16: XRS-B agrees to 0.0004 dex; protons read ~19% lower on GOES-18 during SEPs. C1 (recalibrated Brier skill) is in sweeps/REPORT.md.

## Running now
Nothing. G1 trained 22:44 → 23:57 (3 seeds, 4.8 GB, 1 thread); scoring rerun after a fix, done 00:15.

## Waiting on user
Nothing.

## Next up
Write-up: onboarding notebook cells for C1 and G1 (paste-ready). Optional: check the 2026-01-19 SEP (43,893 pfu on GOES-18) against NOAA's event list.
