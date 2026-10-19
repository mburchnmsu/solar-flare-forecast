# Sweep report 1: physics-informed loss vs. baselines (2026-10-18)

**Question.** Does a physics-informed loss (cooling-time decay bound + quiet-Sun curvature + gradual-phase convexity)
improve a 1D-CNN forecaster of GOES XRS-B and ≥10 MeV proton flux over (a) persistence, (b) the same CNN trained
on MSE only, and (c) the uniform "naive" smoothing penalty from the original prototype?

**Answer (synthetic data).** Yes for X-rays. Over 3 seeds, the physics-loss CNN has the lowest test error, about
14% below persistence and 2% below the MSE-only CNN. That gap is about 6 seed standard deviations. Its active-period
error is 25% below persistence, and its quiet-time forecast roughness is 77% lower than the plain CNN's.
**No for protons:** with one shared λ, the proton forecast collapses to persistence. **No, as it stands, on real data:**
zero-shot on 7 days of live SWPC data, the physics model is 5% *worse* than persistence (plain CNN 2% better).

## Setup

| | |
|---|---|
| Data | `generate_telemetry(TelemetryConfig(seed = 42))`: 90 days at 1-min cadence. 687 flares (454 B, 203 C, 27 M, 3 X), Hawkes-clustered, power-law peaks (α = 1.9), half-Gaussian rise + exponential decay. 5 SEP events. 2% / 6% multiplicative noise; noise-free truth kept for scoring |
| Task | Input: last 180 min of [log XRS-B, log protons, ΔXRS-B]. Output: next 60 min of both channels |
| Split | Chronological 70/15/15 with a W+H = 240-min guard gap between splits; standardiser fit on train only. Train windows stride 3 (30,161), val/test stride 1 (18,961 each) |
| Model | `ResidualForecaster`: 3 conv blocks (16/32/32 ch, GELU, max-pool) → dense 128 → 120 outputs, added to the last observation. Zero-initialised output layer, so the untrained model is persistence |
| Training | Adam 1e-3, clip-norm 1, batch 256, ≤30 epochs, early stopping on val MSE (patience 6), 1 CPU thread per run |
| Selection | One λ per family by **validation** MSE on noisy targets (the only criterion available with real data). Test scored once afterwards, against noise-free truth |
| Seeds | Chosen tags retrained with init/batch-order seeds 1, 2, 3 |

λ grid: naive {0.3, 3, 30}; physics (all three terms share λ) {1e3, 1e4, 1e5, 1e6}. Selected: `naive_30`, `phys_1e4`.

## Results: test split, mean ± std over 3 seeds (dex = log₁₀ flux; RMSE vs noise-free truth)

| Model | X-ray RMSE | X-ray, active | X-ray, quiet | Protons | Quiet roughness (dex/min²) | TSS ≥M (60 min) | Cooling-bound violations |
|---|---|---|---|---|---|---|---|
| Persistence | 0.268 | 0.377 | 0.218 | 0.060 | 0 | 0.391 | 0 |
| CNN (MSE only) | 0.2356 ± 0.0002 | 0.294 ± 0.002 | 0.212 ± 0.001 | **0.057 ± 0.003** | 6.6e-4 ± 0.9e-4 | 0.384 ± 0.010 | 0 |
| CNN + naive smoothing (λ=30) | 0.2354 ± 0.0003 | 0.294 ± 0.003 | 0.212 ± 0.001 | 0.058 ± 0.002 | 4.4e-4 ± 1.5e-4 | 0.308 ± 0.011 | 0.07% |
| **CNN + physics loss (λ=1e4)** | **0.2303 ± 0.0008** | **0.282 ± 0.003** | **0.210 ± 0.000** | 0.060 ± 0.000 | **1.5e-4 ± 1.6e-4** | **0.399 ± 0.006** | 0 |

Noise-free truth: quiet curvature 7.6e-4 dex/min², cooling-bound violations 0 (the bound is physically consistent with
the data). Quiet windows 14,103, active 4,858. Windows with a *new* ≥M crossing (not already ≥M at t₀): 240.

Figures: `figures/skill_vs_horizon.png`, `figures/forecast_examples.png`, `figures/training_curves.png`,
`figures/telemetry_overview.png`. Tables: `results/summary.csv` (seed-aggregated), `results/sweep_table.csv` (per run).

### λ sweep (seed 1, validation MSE)

| λ (physics) | 0 (MSE) | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|
| val MSE | 0.0995 | 0.0977 | **0.0976** | 0.117 | 0.128 (= persistence) |
| epochs to early stop | 12 | 24 | 30 (cap) | 14 | 13 |

The physics terms act as a **regulariser**. The MSE-only model starts overfitting after about 10 epochs, while
λ = 1e3–1e4 keeps improving for 2–3× longer. λ ≥ 1e5 over-constrains the model back to persistence. Naive smoothing
barely moves validation MSE at any λ.

## Zero-shot on real data (SWPC 7-day, 2026-10-12 → 10-19, primary GOES)

Synthetic-trained models and standardiser, no retraining. 9,041 windows; largest flare M1.4; protons quiet (≤0.4 pfu).
Scored against observations.

| Model | X-ray RMSE | active | quiet | Protons |
|---|---|---|---|---|
| Persistence | 0.164 | 0.229 | 0.128 | 0.057 |
| CNN (MSE only) | **0.161** | **0.220** | 0.129 | 0.063 |
| CNN + naive smoothing | **0.161** | **0.219** | 0.129 | 0.064 |
| CNN + physics loss | 0.173 | 0.243 | 0.133 | 0.057 |

`figures/swpc_zero_shot_examples.png` shows why:
* The real M1.4 decays faster and less convexly than the synthetic flares, so the convexity prior makes the physics
  model decay too slowly.
* A 1.6-dex dropout/recovery in the record (probably an eclipse-season or data-quality artefact) gets forecast as a
  flare decay by every model. Real data needs quality-flag filtering.
* In a quiet stretch with an elevated background, the physics model tracks the gentle decline best.

## Findings

1. **The physics loss improves X-ray forecasts on in-distribution data**, mainly in active periods (−4% vs CNN),
   without hurting the quiet periods, and it removes most quiet-time high-frequency structure.
2. **Uniform smoothing is the wrong prior.** At the λ that validation prefers, it costs 20% of M-class TSS
   (0.38 → 0.31) and is the only model that violates the cooling bound. It damps genuine rises and overshoots decays.
3. **The cooling bound never binds for the unconstrained CNN** on synthetic data (0 violations). Its value here is as a
   guard-rail, not a source of skill. The curvature and convexity terms do the work, though the per-term ablation
   hasn't been run yet.
4. **No model forecasts new flare onsets** (TSS on new ≥M crossings ≈ 0.015). That's expected, because onsets in the
   generator are nearly memoryless. Skill comes from evolving events already in progress.
5. **A shared λ is wrong for protons.** The proton terms (3-h decay bound, curvature everywhere) push the proton
   forecast to persistence.
6. **The synthetic→real gap is real.** The priors encode the generator's flare shapes. On real data the plain CNN
   transfers better.

## Open decisions / next experiments (need the user's OK on compute)

* Per-channel λ (protons separately) and a per-term ablation (decay / rough / convex alone at 1e4). [~8 runs, 2 workers, ~2 h]
* Relax or gate the convexity prior by flare-to-background ratio (it's only strictly right when background matters).
* Real-data training on the NCEI GOES-R archive with quality flags; pre-train on synthetic, then fine-tune.
* Probabilistic outputs (quantile loss) and reliability diagrams.
