# Zero-shot test: apply the synthetic-trained models (chosen tags, all seeds found) to real GOES data from SWPC.
# No retraining and no re-fitting of the standardiser, so this measures the synthetic→real domain gap honestly.
#     julia -t auto --project=. sweeps/zero_shot_swpc.jl          # uses every cached download in data/raw/
include(joinpath(@__DIR__, "..", "sol_setup.jl"))
include(joinpath(@__DIR__, "variants.jl"))
using Serialization, Flux

real = load_swpc_cache()
println("SWPC data: ", nrow(real), " min, ", first(real.time), " → ", last(real.time),
        "; peak XRS-B ", flare_class(maximum(real.xray)), "; max ≥10 MeV ", round(maximum(real.proton); digits = 2), " pfu")
_, _, data, _ = sweep_data()
w = window_with(real, data.scaler; W = data.W, H = data.H)
println("Windows: ", length(w))
Yreal = w.Y                                   # no noise-free truth for real data: score against observations

chosen = Dict(split(l)[1] => split(l)[2] for l in eachline(joinpath(RESULTS_DIR, "chosen.txt")))
preds = ["Persistence" => persistence(w)]
for fam in ("mse", "naive", "phys")
    haskey(chosen, fam) || continue
    f = joinpath(@__DIR__, "results", "$(chosen[fam])_s1", "run.jls")
    r = deserialize(f)
    m = build_model(data.W, data.H)
    Flux.loadmodel!(m, r.state)
    push!(preds, FAMILY[fam] => predict(m, w.X))
end

rows = NamedTuple[]
for (n, ŷ) in preds
    e = evaluate(ŷ, w, data.scaler; Yclean = Yreal, loss = PhysicsLoss())
    push!(rows, (; model = n, rmse_x = e.rmse_x, rmse_x_quiet = e.rmse_x_quiet, rmse_x_active = e.rmse_x_active,
                 rmse_p = e.rmse_p, rough_quiet = e.rough_quiet, decay_violation_frac = e.decay_violation_frac,
                 n_quiet = e.n_quiet, n_active = e.n_active))
end
tab = DataFrame(rows)
open(io -> show(io, MIME("text/csv"), tab), joinpath(RESULTS_DIR, "zero_shot_swpc.csv"), "w")
show(stdout, tab; allcols = true, eltypes = false); println()

# example panels: the biggest real flare's decay, its rise, and the quietest stretch
s = data.scaler
now_x = vec(to_log10(s, w.X[end:end, 1, :], 1))
rise = now_x .- vec(to_log10(s, w.X[end-9:end-9, 1, :], 1))
quiet, decaying = regime_gates(w.X, s.σ[1], 0.15f0)
pick(mask, score) = (i = findall(mask); isempty(i) ? argmax(score) : i[argmax(score[i])])
cases = [("SWPC: decay of the largest flare in the window", pick(vec(decaying) .> 0.5, now_x)),
         ("SWPC: fastest rise in the window", argmax(rise)),
         ("SWPC: quietest stretch", pick(vec(quiet) .> 0.9, -now_x)),
         ("SWPC: quiet, elevated background", pick(vec(quiet) .> 0.9, now_x))]
plot_forecast_examples(cases, w, s, preds[2:end], Yreal; truth_label = "Observed (future)", path = joinpath(FIG_DIR, "swpc_zero_shot_examples.png"))
println("Figure → ", joinpath(FIG_DIR, "swpc_zero_shot_examples.png"))
