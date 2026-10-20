# SolarFlareForcast status
Updated: 2026-10-19 21:04, DESKTOP-BBASC81

## Headline
Synthetic sweeps done: the per-channel physics loss is best on both channels (X-ray −14% vs persistence, protons now below persistence), and the ablation shows the quiet-curvature term does the work. A validated real-data set (GOES-16 NCEI, 2022–25) is ready; real-data training is waiting for memory while Zodi runs.

## Best model
`physp_1e4_0.01`: residual 1D-CNN (180 min → 60 min) + physics loss at λ = 1e4, proton penalties × 0.01.
Synthetic test (seed 1): X-ray RMSE 0.2293 dex (persistence 0.268, plain CNN 0.2356), active 0.278, protons 0.056 (persistence 0.060), TSS ≥M 0.40.

## Latest results
Sweep 2, synthetic test split, seed 1, RMSE in dex vs noise-free truth (seed std ≈ 0.0008):

| Variant | X-ray | Protons | Quiet roughness |
|---|---|---|---|
| CNN (MSE), 3 seeds | 0.2356 | 0.057 | 6.6e-4 |
| Physics, shared λ, 3 seeds | 0.2303 | 0.060 | 1.5e-4 |
| Physics, proton × 0.01 | 0.2293 | 0.056 | 3.4e-4 |
| Quiet curvature only | 0.2295 | 0.060 | 2.0e-4 |
| Cooling bound only | 0.2367 | 0.060 | 3.5e-4 |
| Convexity only | 0.2338 | 0.056 | 1.4e-3 |

Real-data loader validation: XRS-B vs SWPC median −0.001 dex; reconstructed SEP peaks vs NOAA 2024 list median ratio 1.19, scatter 0.22 dex (n = 8).

## Running now
Real-data training on GOES-16 (MSE, physp_1e4_0.01, physp fine-tuned from synthetic): 2 workers, started 21:04, expected ~1–2 h.

## Waiting on user
Nothing.

## Next up
Real-data training + analysis (sweeps/train_real.jl, analyze_real.jl), then report, onboarding notebook update, and CV bullets with real-data numbers.
