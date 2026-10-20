# Score every real-data run (sweeps/results/real_*) on the real test split (2024-07 → 2025-03), against
# observations, next to persistence. Each run is evaluated with its own standardiser (fine-tuned runs keep the
# synthetic one), so all errors are in physical dex and directly comparable.
#     julia -t 2 --project=. sweeps/analyze_real.jl
include(joinpath(@__DIR__, "train_real.jl"))           # real_data(), REAL_FILE, variants
using Serialization

runs = Dict{String,Any}()
for d in readdir(joinpath(@__DIR__, "results"); join = true)
    startswith(basename(d), "real_") || continue
    f = joinpath(d, "run.jls")
    isfile(f) && (runs[basename(d)] = deserialize(f))
end
println("Loaded ", length(runs), " real-data runs")

df, data = real_data()
cache = Dict{Any,Any}(nothing => data)
data_for(r) = r.init_run == "-" ? data :
    get!(() -> real_data(; scaler = r.scaler)[2], cache, (r.scaler.μ..., r.scaler.σ...))   # key by value, not identity

const KEYS = (:rmse_x, :rmse_x_quiet, :rmse_x_active, :rmse_p, :rough_quiet, :decay_violation_frac,
              :tss_m, :tss_m_new, :tss_s1)
ev(ŷ, d, split) = evaluate(ŷ, getfield(d, split), d.scaler; Yclean = getfield(d, split).Y, loss = PhysicsLoss())
rows = NamedTuple[]
e = ev(persistence(data.test), data, :test)
push!(rows, (; model = "Persistence", run = "persistence", val_mse = NaN, epochs = 0, (k => getfield(e, k) for k in KEYS)...))
preds = Dict{String,Any}()
for (name, r) in sort(collect(runs); by = first)
    d = data_for(r)
    er = ev(r.pred_test, d, :test)
    # validation MSE in physical units (dex²) so runs with different scalers compare fairly
    vm = mean(abs2, (r.pred_val .- d.val.Y) .* reshape(d.scaler.σ, 1, 2, 1))
    push!(rows, (; model = r.group * (r.init_run == "-" ? "" : " (fine-tuned)"), run = name, val_mse = vm,
                 epochs = nrow(r.history), (k => getfield(er, k) for k in KEYS)...))
    preds[name] = (r, d)
end
t = DataFrame(rows)
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "real_metrics.csv"), "w")
show(stdout, t; allrows = true, allcols = true, eltypes = false); println()
println("Test windows: ", length(data.test), " (stride 5 min)")

# forecast examples on the real test split: largest flare decay, fastest rise, quiet, SEP rise
best = sort(collect(keys(preds)))
if !isempty(best)
    s = data.scaler
    now_x = vec(to_log10(s, data.test.X[end:end, 1, :], 1))
    rise = now_x .- vec(to_log10(s, data.test.X[end-9:end-9, 1, :], 1))
    quiet, decaying = regime_gates(data.test.X, s.σ[1], 0.15f0)
    pick(mask, score) = (i = findall(mask); isempty(i) ? argmax(score) : i[argmax(score[i])])
    i_decay = pick(vec(decaying) .> 0.5, now_x)
    far = abs.(data.test.t0 .- data.test.t0[i_decay]) .> 24 * 60          # a different event for the rise panel
    below_m = now_x .< -5.3                                                # quiet panels: not a flare tail
    cases = [("Real: decay of the largest test flare", i_decay),
             ("Real: fastest rise (different event)", pick(far, rise)),
             ("Real: quiet Sun", pick(vec(quiet) .> 0.9, -now_x)),
             ("Real: quiet, elevated background", pick((vec(quiet) .> 0.9) .& below_m, now_x))]
    # show each family's real-scaler run; fine-tuned predictions are mapped onto the real scaler for plotting
    shown = Pair{String,Array{Float32,3}}[]
    for name in best
        r, d = preds[name]
        fam = r.group in keys(MODEL_COLORS) ? r.group : nothing
        (fam === nothing || r.init_run != "-" || any(first.(shown) .== fam)) && continue
        push!(shown, fam => r.pred_test)
    end
    plot_forecast_examples(cases, data.test, s, shown, data.test.Y; truth_label = "Observed (future)",
                           path = joinpath(FIG_DIR, "real_forecast_examples.png"))
    println("Figure → figures/real_forecast_examples.png")
end
