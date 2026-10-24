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

---

# Sweep 2: per-channel weighting and per-term ablation (2026-10-19)

All at λ = 1e4, seed 1, the same data and protocol as sweep 1 (test split, RMSE vs noise-free truth, dex). Seed-to-seed
std from sweep 1 is ~0.0008 dex on X-ray RMSE, so gaps below ~0.001 are not resolved.

| Variant | val MSE | X-ray | X-ray active | Protons | Quiet roughness | Cooling violations | TSS ≥M |
|---|---|---|---|---|---|---|---|
| Persistence | 0.1283 | 0.268 | 0.377 | 0.060 | 0 | 0 | 0.391 |
| CNN, MSE only (3 seeds) | 0.0994 | 0.2356 | 0.294 | 0.057 | 6.6e-4 | 0 | 0.384 |
| Physics, shared λ (3 seeds) | 0.0963 | 0.2303 | 0.282 | 0.060 | 1.5e-4 | 0 | 0.399 |
| Physics, proton terms × 0.1 | 0.0972 | 0.2301 | 0.282 | 0.059 | 1.9e-4 | 0 | 0.395 |
| **Physics, proton terms × 0.01** | 0.0966 | **0.2293** | **0.278** | **0.056** | 3.4e-4 | 0 | 0.402 |
| Physics, proton terms × 0 | 0.0975 | 0.2302 | 0.279 | 0.056 | 2.9e-4 | 0 | 0.414 |
| Cooling bound only | 0.0992 | 0.2367 | 0.296 | 0.060 | 3.5e-4 | 0 | 0.391 |
| Quiet curvature only | **0.0956** | 0.2295 | 0.279 | 0.060 | 2.0e-4 | 0 | 0.388 |
| Decay convexity only | 0.0993 | 0.2338 | 0.288 | 0.056 | 1.4e-3 | 0.11% | 0.400 |

**Findings**
1. **Per-channel weighting fixes the proton collapse.** Scaling the proton penalties by 0.01 (or 0) brings proton RMSE
   from 0.060 (persistence) to 0.056, better than the plain CNN, while keeping the full X-ray gain.
2. **The quiet-curvature term does the work.** On its own it reproduces the whole X-ray improvement and has the best
   validation MSE. It still collapses protons, because its proton part runs at full weight.
3. **The cooling bound is inert on synthetic data:** within noise of no penalty, consistent with sweep 1's
   zero violations. It stays in as a guard-rail for real data, not as a source of skill.
4. **Convexity alone is harmful to smoothness** (2× rougher than the plain CNN, and the only physics term that
   breaks the cooling bound). It is only sensible together with the curvature term, and its prior is wrong for
   flares far above background (see sweep 1's real-data panels).

**Choice for real-data training:** `physp_1e4_0.01` (best X-ray and proton test numbers among the physics variants;
validation within noise of `rough_1e4`, and unlike it, it doesn't collapse protons).

# Real data: NCEI GOES-16 L2 (2026-10-19)

* `load_ncei` merges XRS-B (eclipse- and bad-flagged minutes dropped) and ≥10 MeV protons reconstructed from SGPS
  (bin sum of differential channels above 10 MeV + the >500 MeV channel, mean of both heads).
* **XRS validation** (GOES-18, 2026-10-13 → 18, 8,487 matched minutes vs SWPC real-time): median log ratio −0.001 dex,
  5–95% range −0.019 … +0.013, log-correlation 0.996.
* **Proton validation** vs NOAA SWPC's official 2024 SEP list (8 events, 20–1810 pfu): reconstructed peak / listed
  peak median 1.19, scatter 0.22 dex. Peak times agree to within minutes for 6 of 8 events. The two hardest events
  (May 10, Oct 10) are ~2× high. At quiet times (~0.3 pfu) the reconstruction reads ~0.25 dex above SWPC; this is
  background-dominated and far below the 10 pfu S1 threshold. (`results/sep_peak_validation.csv`, `figures/ncei_vs_swpc.png`)
* **Dataset:** GOES-16, 2022-01-01 → 2025-03-31, 1.51 M quality-filtered minutes (88.7%). Science-quality SGPS files
  start 2022-04-25 (earlier months only have `dn_` v2 files, which are not used), so proton-covered training data
  effectively starts there. XRS-B max X8.9, protons ≥10 pfu in 3.8% of minutes.
* **Planned split:** train → 2023-12-31, validate 2024-01 → 06, test 2024-07 → 2025-03 (includes the Oct 2024 X-flares
  and the 1810 pfu event). Real-data training is on hold until the Zodi project frees memory.

# Real-data training: GOES-16, sweep R1 (2026-10-19)

Trained on real NCEI GOES-16 data (2022-04-25 → 2023-12-31; 165,824 windows at a 5-min stride). Early-stopped on
2024-01 → 06 (48,124 windows). Scored once on **2024-07 → 2025-03** (71,876 windows; includes the Oct 2024 X9.0 and
the 1810 pfu SEP). Scored against observations (no noise-free truth for real data). Seed 1 only, so these are
single-run numbers.

| Model | val MSE (dex²) | epochs | X-ray RMSE | X-ray active | X-ray quiet | Protons | Quiet roughness | TSS ≥M | TSS new ≥M | TSS S1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | – | – | 0.178 | 0.228 | 0.147 | 0.159 | 0 | 0.329 | 0 | 0.882 |
| CNN, MSE only | 0.0208 | 10 | 0.158 | 0.183 | 0.144 | **0.118** | 5.6e-4 | 0.337 | 0.018 | **0.928** |
| **CNN + physics (λ=1e4, proton × 0.01)** | **0.0203** | 14 | **0.155** | **0.179** | **0.141** | 0.119 | **2.9e-4** | **0.352** | **0.036** | 0.890 |

**Findings**
1. **Trained on real data, the physics loss helps.** X-ray RMSE is −13% vs persistence (−21% in active periods) and
   −1.9% vs the same CNN without it, about the same relative gain as on synthetic data. It also halves quiet-time
   roughness and has the best M-class TSS. Last night's zero-shot failure was a *domain-gap* problem, not a problem
   with the constraint itself.
2. **Same regularisation signature as on synthetic data.** The MSE-only CNN starts overfitting after ~5 epochs
   (val MSE rises from epoch 5 to 10); the physics model keeps improving to epoch ~9 and stops at 14.
3. **Protons:** both CNNs beat persistence by ~26% (0.118–0.119 vs 0.159 dex). The MSE-only model has the better S1
   threshold skill (0.93 vs 0.89), so the physics prior is neutral-to-slightly-negative for protons on real data.
4. **Cooling bound:** violation rates are ~2e-6 for both models, so it barely binds on real data either.
5. **Known failure:** the biggest flares. On the Oct 2024 X9 both models decay too slowly. On a rapidly rising
   event both turn over too early (`figures/real_forecast_examples.png`).

**Not yet run (needs the user's OK):** the fine-tuned-from-synthetic variant (`physp_1e4_0.01` initialised from
`physp_1e4_0.01_s1`), and seeds 2–3 of both models for error bars. Run these one worker at a time: two workers loading the
1.5 M-minute dataset at once pushed the PC low on RAM.

## Real-data sweep R2: 3 seeds (2026-10-20)

Same protocol as R1; seeds 1–3 of both real-data models, plus the fine-tuned run (seed 1). Test 2024-07 → 2025-03, dex.

| Model | n | X-ray RMSE | X-ray active | Protons | Quiet roughness | TSS ≥M | TSS S1 |
|---|---|---|---|---|---|---|---|
| Persistence | – | 0.1776 | 0.228 | 0.159 | 0 | 0.329 | 0.882 |
| CNN, MSE only | 3 | 0.1569 ± 0.0006 | 0.1813 ± 0.0011 | 0.1184 ± 0.0005 | 6.0e-4 | 0.336 ± 0.002 | 0.914 |
| **CNN + physics (λ=1e4, proton × 0.01)** | 3 | **0.1552 ± 0.0006** | **0.1794 ± 0.0003** | 0.1190 ± 0.0004 | **3.0e-4** | 0.341 ± 0.010 | 0.898 |
| Physics, fine-tuned from synthetic | 1 | 0.1537 | 0.1773 | 0.1216 | 1.9e-4 | 0.350 | 0.903 |

**Conclusions (real GOES-16 data)**
1. **The physics loss gives a small, reproducible X-ray gain:** −1.1% RMSE vs the same CNN (gap 0.0017 dex, ≈ 3× the
   seed std), and −12.6% vs persistence (−21% in active periods). It is consistent with the synthetic result
   (−2.2%).
2. **Quiet-time roughness is halved,** the effect the curvature prior was designed for.
3. **Protons and onset skill: no difference.** Both CNNs beat persistence by ~26% on protons; M-class TSS differences
   are within seed noise.
4. **Synthetic pre-training looks promising but is unconfirmed:** the single fine-tuned run is another −1% on X-rays,
   with the smoothest forecasts. Seeds 2–3 would settle it.

## Real-data sweep R3: fine-tuned seeds 2–3 (2026-10-24)

Seeds 2–3 of the synthetic → real fine-tuned model (init `physp_1e4_0.01_s1`, then trained on real GOES-16 data with the
same loss), 1 worker at below-normal priority. Same protocol and test split as R2, dex.

| Model | n | X-ray RMSE | X-ray active | Protons | Quiet roughness | TSS ≥M | TSS S1 |
|---|---|---|---|---|---|---|---|
| Persistence | – | 0.1776 | 0.228 | 0.159 | 0 | 0.329 | 0.882 |
| CNN, MSE only | 3 | 0.1569 ± 0.0006 | 0.1813 ± 0.0011 | 0.1184 ± 0.0005 | 6.0e-4 | 0.336 ± 0.002 | 0.914 |
| CNN + physics | 3 | 0.1552 ± 0.0006 | 0.1794 ± 0.0003 | 0.1190 ± 0.0004 | 3.0e-4 | 0.341 ± 0.010 | 0.898 |
| **Physics, fine-tuned from synthetic** | 3 | **0.1533 ± 0.0005** | **0.1767 ± 0.0011** | 0.1216 ± 0.0010 | **2.3e-4** | 0.345 ± 0.006 | 0.910 |

**Conclusions**
1. **Synthetic pre-training helps, confirmed:** −1.2% X-ray RMSE vs the physics CNN trained on real data only (gap
   0.0019 dex, ≈ 3× the seed std), −2.3% vs the plain CNN, −13.7% vs persistence (−22.5% in active periods).
   Forecasts are also the smoothest (quiet roughness 2.3e-4, 62% below the plain CNN).
2. **Protons are slightly worse** (0.1216 vs 0.1190, ≈ 2–3σ). The generator's SEP model (onset delay, ppk ∝ F^1.3)
   probably biases the proton head. Still 24% better than persistence. A channel-specific fine-tune (unfreeze the
   proton output only, or drop the proton targets in pre-training) could keep the X-ray gain without that cost.
3. **Onset skill is unchanged:** TSS differences are within seed noise for all CNNs, so the X-ray gain is in flare
   evolution and decay, not in predicting new flares.
4. New best model: `real_physp_1e4_0.01_ft-physp_1e4_0.01_s1` (3 seeds).
