# End-to-end run: synthetic telemetry → windows → three CNN variants + persistence → metrics, figures.
#
#     julia -t auto --project=. scripts/run_pipeline.jl            # full run (~35 min on 8 CPU threads)
#     julia -t auto --project=. scripts/run_pipeline.jl --quick    # 3 epochs per model, smoke test
#
# Single-process demo of the whole pipeline; the reported numbers come from sweeps/ (multi-seed, λ selected on val).
# Writes figures/demo_*.png, results/demo_metrics.csv, results/demo_skill_by_horizon.csv, results/demo_history_*.csv,
# and results/demo_models.jls (Flux states, reload with Flux.loadmodel!).

include(joinpath(@__DIR__, "..", "sol_setup.jl"))
using Serialization, Flux

const QUICK = "--quick" in ARGS
const EPOCHS = QUICK ? 3 : 30

# ---------------------------------------------------------------- data
df, events = generate_telemetry(TelemetryConfig(seed = 42))
println("Telemetry: ", nrow(df), " min, ", nrow(events), " flares (",
        count(>=(1e-5), events.peak_flux), " ≥M1), ", count(events.sep), " SEP events")
data = prepare_data(df; W = 180, H = 60)
println("Windows: train ", length(data.train), ", val ", length(data.val), ", test ", length(data.test))
Zclean = standardize(data.scaler, log_features(df; xray = :xray_clean, proton = :proton_clean))
Yclean_test = targets_from(Zclean, data.test)
plot_telemetry(df, data.rows; path = joinpath(FIG_DIR, "demo_telemetry_overview.png"))

# ---------------------------------------------------------------- models
variants = [
    "CNN (MSE only)" => PhysicsLoss(),
    "CNN + naive smoothing" => PhysicsLoss(λ_naive = 30f0),          # λ chosen in sweeps/REPORT.md
    "CNN + physics loss" => PhysicsLoss(λ_decay = 1f4, λ_rough = 1f4, λ_convex = 1f4),
]
trained = Pair{String,Any}[]
histories = Pair{String,DataFrame}[]
for (name, loss) in variants
    model, hist = train_model(data, loss; epochs = EPOCHS, seed = 1, label = name)
    push!(trained, name => model)
    push!(histories, name => hist)
    CSV_name = replace(lowercase(name), r"[^a-z]+" => "_")
    open(io -> show(io, MIME("text/csv"), hist), joinpath(RESULTS_DIR, "demo_history_$(strip(CSV_name, '_')).csv"), "w")
end
serialize(joinpath(RESULTS_DIR, "demo_models.jls"), Dict(n => Flux.state(m) for (n, m) in trained))

# ---------------------------------------------------------------- evaluation (test split only)
reference_loss = last(variants).second             # thresholds/gates used for the diagnostics
preds = ["Persistence" => persistence(data.test)]
append!(preds, [n => predict(m, data.test.X) for (n, m) in trained])
results = [n => evaluate(ŷ, data.test, data.scaler; Yclean = Yclean_test, loss = reference_loss) for (n, ŷ) in preds]

scalar_keys = (:rmse_x, :rmse_x_clean, :rmse_x_quiet, :rmse_x_active, :rmse_p, :rmse_p_clean,
               :rough_quiet, :decay_violation_frac, :tss_m, :tss_m_new, :tss_s1)
metrics = DataFrame([(; model = n, (k => getfield(r, k) for k in scalar_keys)...) for (n, r) in results])
open(io -> show(io, MIME("text/csv"), metrics), joinpath(RESULTS_DIR, "demo_metrics.csv"), "w")
horizon = DataFrame(lead_min = 1:data.H)
for (n, r) in results
    horizon[!, "x_" * n] = r.rmse_x_h
    horizon[!, "p_" * n] = r.rmse_p_h
end
open(io -> show(io, MIME("text/csv"), horizon), joinpath(RESULTS_DIR, "demo_skill_by_horizon.csv"), "w")

println("\nTest-split metrics (dex; 'clean' = vs noise-free truth). Truth quiet roughness: ",
        @sprintf("%.2e", last(results).second.rough_quiet_truth), " dex/min²; quiet windows: ",
        last(results).second.n_quiet, ", active: ", last(results).second.n_active)
show(stdout, metrics; allrows = true, allcols = true, eltypes = false)
println()

# ---------------------------------------------------------------- figures
plot_training(histories; path = joinpath(FIG_DIR, "demo_training_curves.png"))
plot_skill_vs_horizon(results; path = joinpath(FIG_DIR, "demo_skill_vs_horizon.png"))

# Pick illustrative test windows deterministically: biggest decaying flare, fastest rise, quietest, and a
# moderately active background.
s = data.scaler
now_x = vec(to_log10(s, data.test.X[end:end, 1, :], 1))
quiet, decaying = regime_gates(data.test.X, s.σ[1], reference_loss.quiet_dex)
rise = vec(to_log10(s, data.test.X[end:end, 1, :], 1) .- to_log10(s, data.test.X[end-9:end-9, 1, :], 1))
pick(mask, score) = (i = findall(mask); i[argmax(score[i])])
cases = [
    "Gradual (decay) phase of the largest test flare" => pick(vec(decaying) .> 0.5, now_x),
    "Impulsive rise in progress" => argmax(rise),
    "Quiet Sun, low background" => pick(vec(quiet) .> 0.9, -now_x),
    "Quiet but elevated (C-level) background" => pick((vec(quiet) .> 0.9) .& (now_x .< -5.7), now_x),
]
plot_forecast_examples([(t, n) for (t, n) in cases], data.test, s, preds[2:end], Yclean_test;
                       path = joinpath(FIG_DIR, "demo_forecast_examples.png"))
println("Figures written to ", FIG_DIR)
