# Collect every run in sweeps/results, select one λ per loss family on the VALIDATION split (noisy-target MSE —
# the only criterion available with real data), then report test metrics as mean ± std over init seeds.
#     julia -t auto --project=. sweeps/analyze.jl            # tables + figures
# Writes results/sweep_table.csv (one row per run), results/summary.csv (per tag, seed-aggregated),
# results/chosen.txt (selected tag per family), results/skill_by_horizon.csv, figures/*.png.
include(joinpath(@__DIR__, "..", "sol_setup.jl"))
include(joinpath(@__DIR__, "variants.jl"))
using Serialization

df, events, data, Zclean = sweep_data()
gates = PhysicsLoss()                                    # one set of regime gates/bounds for every model
Yclean = (val = targets_from(Zclean, data.val), test = targets_from(Zclean, data.test))
ev(ŷ, split) = evaluate(ŷ, getfield(data, split), data.scaler; Yclean = getfield(Yclean, split), loss = gates)

const KEYS = (:rmse_x, :rmse_x_clean, :rmse_x_quiet, :rmse_x_active, :rmse_p, :rmse_p_clean,
              :rough_quiet, :decay_violation_frac, :tss_m, :tss_m_new, :tss_s1)

runs = Dict{String,Any}()
for d in readdir(joinpath(@__DIR__, "results"); join = true)
    f = joinpath(d, "run.jls")
    isfile(f) && (runs[basename(d)] = deserialize(f))
end
println("Loaded ", length(runs), " runs")

# ---- per-run table ------------------------------------------------------------------------------------
rows = NamedTuple[]
pers = persistence(data.test)
rp = ev(pers, :test)
push!(rows, (; model = "Persistence", tag = "persistence", seed = 0, epochs = 0,
             val_mse = Float64(mean(abs2, persistence(data.val) .- data.val.Y)), (k => getfield(rp, k) for k in KEYS)...))
test_eval = Dict{String,Any}("persistence" => rp)
for (name, r) in sort(collect(runs); by = first)
    e = ev(r.pred_test, :test)
    test_eval[name] = e
    push!(rows, (; model = r.group, tag = r.tag, seed = r.seed, epochs = nrow(r.history),
                 val_mse = Float64(mean(abs2, r.pred_val .- data.val.Y)), (k => getfield(e, k) for k in KEYS)...))
end
sweep = DataFrame(rows)
open(io -> show(io, MIME("text/csv"), sweep), joinpath(RESULTS_DIR, "sweep_table.csv"), "w")

# ---- seed-aggregated summary -----------------------------------------------------------------------------
summary = combine(groupby(sweep, [:model, :tag]), nrow => :n_seeds, :val_mse => mean => :val_mse,
                  [k => mean => k for k in KEYS]..., [k => (v -> length(v) > 1 ? std(v) : NaN) => Symbol(k, :_sd) for k in KEYS]...)
open(io -> show(io, MIME("text/csv"), summary), joinpath(RESULTS_DIR, "summary.csv"), "w")
show(stdout, select(summary, :model, :tag, :n_seeds, :val_mse, :rmse_x_clean, :rmse_x_clean_sd, :rmse_x_quiet,
                    :rmse_x_active, :rmse_p_clean, :rough_quiet, :decay_violation_frac, :tss_m, :tss_m_new, :tss_s1);
     allrows = true, allcols = true, eltypes = false)
println()

# ---- selection on validation (seed-1 runs, mean over seeds if more exist) ---------------------------------
chosen = Dict{String,String}()
for fam in ("mse", "naive", "phys")
    g = FAMILY[fam]
    cand = filter(r -> r.model == g, summary)
    nrow(cand) == 0 && continue
    chosen[fam] = cand.tag[argmin(cand.val_mse)]
    println("Selected for '", g, "': ", chosen[fam], " (val MSE ", round(minimum(cand.val_mse); sigdigits = 5), ")")
end
open(joinpath(RESULTS_DIR, "chosen.txt"), "w") do io
    for fam in ("mse", "naive", "phys")
        haskey(chosen, fam) && println(io, fam, " ", chosen[fam])
    end
end

truth = rp
@printf("\nTruth (noise-free) quiet curvature %.2e dex/min², truth cooling-bound violations %.5f\n",
        truth.rough_quiet_truth, truth.decay_violation_truth)
println("Test windows: quiet ", truth.n_quiet, ", active ", truth.n_active, "; new ≥M windows ", truth.n_m_new_events)

# ---- figures (seed-1 runs of the chosen tags) -----------------------------------------------------------
fam_order = [f for f in ("mse", "naive", "phys") if haskey(chosen, f)]
sel = [FAMILY[f] => runs["$(chosen[f])_s1"] for f in fam_order]
results = ["Persistence" => rp; [n => test_eval["$(r.tag)_s1"] for (n, r) in sel]]

horizon = DataFrame(lead_min = 1:data.H)
for (n, r) in results
    horizon[!, "x_" * n] = r.rmse_x_h
    horizon[!, "p_" * n] = r.rmse_p_h
end
open(io -> show(io, MIME("text/csv"), horizon), joinpath(RESULTS_DIR, "skill_by_horizon.csv"), "w")

plot_telemetry(df, data.rows; path = joinpath(FIG_DIR, "telemetry_overview.png"))
plot_training([n => r.history for (n, r) in sel]; path = joinpath(FIG_DIR, "training_curves.png"))
plot_skill_vs_horizon(results; path = joinpath(FIG_DIR, "skill_vs_horizon.png"))

s = data.scaler
now_x = vec(to_log10(s, data.test.X[end:end, 1, :], 1))
quiet, decaying = regime_gates(data.test.X, s.σ[1], gates.quiet_dex)
rise = now_x .- vec(to_log10(s, data.test.X[end-9:end-9, 1, :], 1))
future_jump = vec(maximum(to_log10(s, Yclean.test[1:20, 1, :], 1); dims = 1)) .- now_x
pick(mask, score) = (i = findall(mask); i[argmax(score[i])])
cases = [
    ("Gradual (decay) phase of the largest test flare", pick(vec(decaying) .> 0.5, now_x)),
    ("Impulsive rise in progress", argmax(rise)),
    ("Quiet Sun, low background", pick(vec(quiet) .> 0.9, -now_x)),
    # an onset the inputs give no hint of: shows what *no* model here can forecast
    ("New flare starting just after forecast time", pick(vec(quiet) .> 0.9, future_jump)),
]
plot_forecast_examples(cases, data.test, s, [n => r.pred_test for (n, r) in sel], Yclean.test;
                       path = joinpath(FIG_DIR, "forecast_examples.png"))
println("Figures → ", FIG_DIR)
