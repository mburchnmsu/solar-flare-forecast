# SolarFlareForcast status
Updated: 2026-10-20 16:53, DESKTOP-BBASC81

## Headline
First real-data result: trained on GOES-16 (2022–23) and tested on Jul 2024 – Mar 2025, the physics-loss CNN beats persistence by 13% on X-ray flux (21% during flares) and the plain CNN by 2%, and halves quiet-time roughness. Single seed so far.

## Best model
`real_physp_1e4_0.01_s1`: residual 1D-CNN (180 min → 60 min), physics loss λ = 1e4 with proton terms × 0.01, trained on real GOES-16 NCEI L2 data.
Real test: X-ray RMSE 0.155 dex (persistence 0.178, plain CNN 0.158); active 0.179 (persistence 0.228); protons 0.119 (persistence 0.159); TSS ≥M 0.35.

## Latest results
Real GOES-16 test split (2024-07 → 2025-03, 71,876 windows), seed 1, RMSE in dex vs observations:

| Model | X-ray | X-ray active | Protons | Quiet roughness | TSS ≥M | TSS S1 |
|---|---|---|---|---|---|---|
| Persistence | 0.178 | 0.228 | 0.159 | 0 | 0.33 | 0.88 |
| CNN (MSE) | 0.158 | 0.183 | 0.118 | 5.6e-4 | 0.34 | 0.93 |
| CNN + physics (per-channel) | 0.155 | 0.179 | 0.119 | 2.9e-4 | 0.35 | 0.89 |

## Running now
Seeds 2–3 of the two real-data models (4 runs, 1 worker, ~2.5 GB RAM), started 16:57, expected ~19:00; then analysis. Fine-tuned-from-synthetic run analysed: X-ray 0.1537 dex (vs 0.1545 without pre-training), lowest quiet roughness (1.9e-4).

## Waiting on user
Nothing.

## Next up
Error bars on the real-data result, then report + onboarding notebook + CV bullets with real-data numbers. Details: sweeps/REPORT.md.
