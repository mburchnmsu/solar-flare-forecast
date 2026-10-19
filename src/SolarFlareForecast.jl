"""
    SolarFlareForecast

Physics-constrained forecasting of GOES X-ray flux and ≥10 MeV proton (SEP) flux with a 1D-CNN.
Synthetic telemetry → sliding windows → CNN with physics-informed loss → diagnostics.
"""
module SolarFlareForecast

using CSV, DataFrames, Dates, Downloads, JSON3, Printf, Random, Statistics
using Flux
using ChainRulesCore: ChainRulesCore
using CairoMakie

include("synthetic.jl")
include("ingest.jl")
include("windows.jl")
include("model.jl")
include("train.jl")
include("metrics.jl")
include("plots.jl")

export TelemetryConfig, generate_telemetry, flare_class, XRAY_CLASS_FLOOR, SEP_THRESHOLD_PFU
export load_swpc, load_swpc_cache, read_telemetry_csv, fetch_swpc_json
export Standardizer, log_features, standardize, to_log10, Windows, prepare_data, window_with
export ResidualForecaster, build_model, PhysicsLoss, physics_terms, total_loss, regime_gates
export train_model, predict, persistence, targets_from, evaluate, tss
export plot_telemetry, plot_training, plot_skill_vs_horizon, plot_forecast_examples, MODEL_COLORS

end
