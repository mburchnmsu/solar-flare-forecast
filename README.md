# Physics-constrained solar flare & SEP forecasting (Julia)

A 1D convolutional network forecasts the next **60 minutes** of GOES **XRS-B soft X-ray flux** and **≥10 MeV proton
flux** from the previous 3 hours, at 1-minute cadence. It is trained with a **physics-informed loss** that encodes
flare thermodynamics:

* **Cooling-time decay bound (one-sided):** log flux may not fall faster than the fastest plasma cooling allows.
  Rises stay unpenalised, because the impulsive phase can be arbitrarily fast.
* **Quiet-Sun curvature penalty:** gated on the input window, so it removes high-frequency artefacts on a quiet
  background without damping real flares.
* **Gradual-phase convexity:** exponential cooling on top of a background is convex in log space.

Built with Julia 1.10+ / Flux.jl / DataFrames.jl / CairoMakie.jl / NCDatasets.jl. Fully reproducible from `Project.toml` + `Manifest.toml`.

## Results on real GOES-16 data (NCEI L2; train 2022-04 → 2023-12, test 2024-07 → 2025-03; 3 seeds)

| Model | X-ray RMSE (dex) | X-ray, active periods | ≥10 MeV protons | Quiet-time roughness (dex/min²) |
|---|---|---|---|---|
| Persistence (operational baseline) | 0.178 | 0.228 | 0.159 | 0 |
| CNN, MSE only | 0.1569 ± 0.0006 | 0.181 ± 0.001 | 0.118 | 6.0e-4 |
| CNN + physics loss | 0.1552 ± 0.0006 | 0.179 ± 0.000 | 0.119 | 3.0e-4 |
| CNN + physics loss, pre-trained on synthetic | 0.1533 ± 0.0005 | 0.177 ± 0.001 | 0.122 | **2.3e-4** |
| **…, proton output head reset before fine-tuning** | **0.1530 ± 0.0006** | **0.176 ± 0.001** | 0.121 | 2.8e-4 |

On real data the physics loss cuts X-ray error by 12.6% vs persistence (21% during flares) and by 1.1% vs the same
network without it (≈3× the seed spread), and halves quiet-time roughness. Pre-training on the synthetic generator
and then fine-tuning on real data gains another 1.2% (−13.7% vs persistence, −22.5% during flares) and gives the
smoothest forecasts, at a small cost on protons. Resetting the proton output head before fine-tuning keeps the X-ray
gain (−13.8% vs persistence, −22.8% during flares) and recovers ~40% of that proton cost; it is the best model on
validation. All CNNs beat persistence on protons by 24–26%. The test period includes the Oct 2024 X9.0 flare and the 1810 pfu SEP event. The real-data loader is validated
against NOAA: XRS-B matches SWPC real-time to 0.001 dex (median), and the ≥10 MeV integral reconstructed from the SGPS
differential channels matches the peaks in NOAA's 2024 SEP list with a median ratio of 1.19 (n = 8).

### Synthetic development data (noise-free truth available; 3 seeds)

| Model | X-ray RMSE (dex) | X-ray, active periods | Quiet-time roughness | TSS, ≥M within 60 min |
|---|---|---|---|---|
| Persistence | 0.268 | 0.377 | 0 | 0.391 |
| CNN, MSE only | 0.2356 ± 0.0002 | 0.294 ± 0.002 | 6.6e-4 | 0.384 ± 0.010 |
| CNN + uniform smoothing | 0.2354 ± 0.0003 | 0.294 ± 0.003 | 4.4e-4 | 0.308 ± 0.011 |
| **CNN + physics loss** | **0.2303 ± 0.0008** | **0.282 ± 0.003** | **1.5e-4** | **0.399 ± 0.006** |

Uniform smoothing loses 20% of M-class skill because it can't tell a flare rise from noise. An ablation shows the
regime-gated quiet-curvature term carries the gain, while the cooling bound never binds. Synthetic-trained models
applied zero-shot to real data did *worse* than persistence; training on real data fixed that. Known limits: no model
forecasts brand-new flare onsets, and the largest flares (X9) are decayed too slowly. Full analysis:
[`sweeps/REPORT.md`](sweeps/REPORT.md).

![real-data forecast examples](figures/real_forecast_examples.png)

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'                       # unit tests
julia -t auto --project=. scripts/run_pipeline.jl --quick           # end-to-end smoke test, ~5 min
```

In VS Code or Jupyter, open `SolOnboarding.ipynb` (orientation: the data, the physics, the method, results, future
work) or start a dated notebook in `notebooks/` with `include(joinpath(@__DIR__, "..", "sol_setup.jl"))`.

Reproduce the reported numbers (λ sweep, then seeds 2–3 of the chosen models, then analysis):

```bash
printf "mse 1\nnaive_30 1\nphys_1e4 1\n" | bash sweeps/run_pool.sh 2    # one line per "tag seed"
julia -t 2 --project=. sweeps/analyze.jl                                # selection on val, test tables, figures
julia -t 2 --project=. sweeps/zero_shot_swpc.jl                         # apply to cached real SWPC data
```

## Real data

`load_swpc()` downloads SWPC's live 7-day GOES products (`xrays-7-day.json`, `integral-protons-7-day.json`), puts both
channels on one 1-minute grid (short gaps interpolated in log space, long gaps dropped), and returns the same schema as
the synthetic generator: `time, minute, xray [W m⁻²], proton [pfu]`. Every download is cached in `data/raw/`, and
`load_swpc_cache()` merges all cached files, so running `load_swpc()` weekly builds a continuous history. Windowing
never crosses a data gap. `data/raw/` keeps one SWPC snapshot (2026-10-12 → 10-19; NOAA data, public domain) so
the zero-shot result reproduces offline.

For multi-year training sets, `load_ncei(16, Date(2022,4,25), Date(2025,3,31))` downloads NOAA's science-quality
NCEI GOES-R L2 archive (XRS netCDF with quality flags; SGPS differential channels summed to a ≥10 MeV integral)
into the same schema, cached in `data/ncei/` (not committed). `read_telemetry_csv` loads any other source with
the same columns.

## Layout

```
src/synthetic.jl   GOES-like generator: Hawkes-clustered power-law flares, SEPs with transport delay, noise + clean truth
src/ingest.jl      SWPC JSON download/cache/merge, 1-min regridding, CSV loader
src/ncei.jl        NCEI GOES-R L2 XRS + SGPS netCDF download, quality flags, ≥10 MeV reconstruction
src/windows.jl     log transform, train-only standardisation, contiguous windows, chronological split with guard gaps
src/model.jl       ResidualForecaster (CNN + persistence skip) and PhysicsLoss
src/train.jl       Adam + clip-norm, early stopping on validation MSE, seeded RNG everywhere
src/metrics.jl     RMSE by regime and lead time, roughness, cooling-bound violations, TSS
src/plots.jl       CairoMakie diagnostics
sweeps/            experiment scripts, REPORT.md (results/ is git-ignored)
notebooks/         dated working notebooks (SolMMDDYYYY.ipynb)
```

## Design notes

* **Persistence is the bar.** The network predicts a *correction* to the last observation, and its output layer
  starts at zero, so an untrained model is exactly persistence.
* **No leakage.** The split is chronological with a guard gap of W + H minutes between splits; the standardiser is fit
  on training rows only; λ is chosen on validation; the test split is scored once.
* **Scored against truth.** The generator keeps noise-free light curves, so smoothness can be told apart from bias.
* **Type-stable core.** Float32 arrays end to end, concrete struct fields, `@inbounds` loops in the generator and
  windowing, explicit broadcasting.
