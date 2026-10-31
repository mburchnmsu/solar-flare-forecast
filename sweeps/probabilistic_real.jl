# Probabilistic forecasts on REAL GOES-16 data: a quantile head (5/25/50/75/95%) on the frozen trunk of each seed of
# the best point model, compared with static error bars on persistence and on the same point model.
#     julia -t 1 --project=. sweeps/probabilistic_real.jl [base_config]
# base_config defaults to the R4 best model; its seeds _s1.._s3 must exist in sweeps/results.
# Protocol: head trained on train, early-stopped on validation; static error bars fit on validation residuals
# (per lead time, channel and quiet/active regime); test scored once. Event base rates for Brier skill come from
# validation. Output: results/prob_metrics.csv, results/prob_summary.csv, results/prob_reliability.csv,
# figures/prob_reliability.png, figures/prob_examples.png, sweeps/results/prob_<config>_s<seed>/run.jls
include(joinpath(@__DIR__, "train_real.jl"))           # real_data(), sol_setup
using Serialization, Flux

base = length(ARGS) > 0 ? ARGS[1] : "real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr"
runs = [deserialize(joinpath(@__DIR__, "results", "$(base)_s$(k)", "run.jls")) for k in 1:3]
scalers = unique((r.scaler.μ..., r.scaler.σ...) for r in runs)
length(scalers) == 1 || error("seeds of $base use different standardisers")
df, data = runs[1].init_run == "-" ? real_data() : real_data(; scaler = runs[1].scaler)
s, σx = data.scaler, data.scaler.σ[1]
println(base, ": train ", length(data.train), ", val ", length(data.val), ", test ", length(data.test), " windows")
τ = QUANTILE_LEVELS

logy(Y, c) = Y[:, c, :] .* s.σ[c] .+ s.μ[c]
clim_m = mean(vec(maximum(logy(data.val.Y, 1); dims = 1)) .>= -5)
clim_s1 = mean(vec(maximum(logy(data.val.Y, 2); dims = 1)) .>= 1)
@printf("validation base rates: ≥M1 within 60 min %.4f, ≥S1 within 60 min %.4f\n", clim_m, clim_s1)

const DROP = (:prob_m, :obs_m, :prob_s1, :obs_s1)
scalars(r) = NamedTuple{filter(k -> !(k in DROP), keys(r))}(r)

# persistence + static error bars (identical for every seed)
pers_val, pers_test = persistence(data.val), persistence(data.test)
emp_pers = EmpiricalQuantiles(pers_val, data.val.Y, data.val.X, σx)
sc_pers = score_quantiles(apply_quantiles(emp_pers, pers_test, data.test.X, σx), data.test, s; clim_m, clim_s1)
rows = NamedTuple[(; model = "Persistence + static error bars", seed = 0, scalars(sc_pers)...)]
probs = Dict{String,Any}("Persistence + static error bars" => sc_pers)
examples = Dict{Int,Any}()

# forecast-example windows: largest flare decay, fastest rise, quiet (same rules as analyze_real.jl)
now_x = vec(to_log10(s, data.test.X[end:end, 1, :], 1))
rise = now_x .- vec(to_log10(s, data.test.X[end-9:end-9, 1, :], 1))
quiet, decaying = regime_gates(data.test.X, σx, 0.15f0)
pick(mask, score) = (i = findall(mask); isempty(i) ? argmax(score) : i[argmax(score[i])])
i_decay = pick(vec(decaying) .> 0.5, now_x)
i_rise = pick(abs.(data.test.t0 .- data.test.t0[i_decay]) .> 24 * 60, rise)
i_quiet = pick((vec(quiet) .> 0.9) .& (now_x .< -5.3), now_x)
cases = [("Decay of the largest test flare", i_decay), ("Fastest rise (different event)", i_rise),
         ("Quiet, elevated background", i_quiet)]

for (k, r) in enumerate(runs)
    model = build_model(data.W, data.H)
    Flux.loadmodel!(model, r.state)
    Ftr, Ptr = trunk_features(model, data.train.X)
    Fva, Pva = trunk_features(model, data.val.X)
    Fte, Pte = trunk_features(model, data.test.X)
    maximum(abs, Pte .- r.pred_test) < 1f-4 || @warn "point forecasts differ from the saved run" maximum(abs, Pte .- r.pred_test)

    emp_pt = EmpiricalQuantiles(Pva, data.val.Y, data.val.X, σx)
    sc_pt = score_quantiles(apply_quantiles(emp_pt, Pte, data.test.X, σx), data.test, s; clim_m, clim_s1)
    push!(rows, (; model = "Best model + static error bars", seed = k, scalars(sc_pt)...))

    head, hist = train_quantile_head(Ftr, Ptr, data.train.Y, Fva, Pva, data.val.Y; epochs = 60, patience = 6,
                                     seed = k, label = "quantile head s$k")
    qte = predict_quantiles(head, Fte, Pte)
    sc_q = score_quantiles(qte, data.test, s; clim_m, clim_s1)
    push!(rows, (; model = "Best model + learned quantile head", seed = k, scalars(sc_q)...))
    @printf("seed %d  pinball X: head %.5f  static %.5f  persistence %.5f\n", k, sc_q.pinball_x, sc_pt.pinball_x, sc_pers.pinball_x)

    out = joinpath(@__DIR__, "results", "prob_$(base)_s$(k)")
    mkpath(out)
    serialize(joinpath(out, "run.jls"), (; base_run = "$(base)_s$(k)", seed = k, τ, head = Flux.state(head),
                                          history = hist, emp_point = emp_pt, emp_persistence = emp_pers))
    if k == 1
        probs["Best model + static error bars"] = sc_pt
        probs["Best model + learned quantile head"] = sc_q
        qst = apply_quantiles(emp_pt, Pte, data.test.X, σx)
        for (_, n) in cases
            examples[n] = (; head = qte[:, 1, :, n], static = qst[:, 1, :, n], point = Pte[:, 1, n])
        end
    end
end

t = DataFrame(rows)
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "prob_metrics.csv"), "w")
show(stdout, t; allrows = true, allcols = true, eltypes = false); println()
keys_ = filter(k -> !(k in (:model, :seed)), propertynames(t))
summary = combine(groupby(t, :model), nrow => :n_seeds, [k => mean => k for k in keys_]...,
                  [k => (v -> length(v) > 1 ? std(v) : NaN) => Symbol(k, :_sd) for k in keys_]...)
open(io -> show(io, MIME("text/csv"), summary), joinpath(RESULTS_DIR, "prob_summary.csv"), "w")
for r in eachrow(summary)
    @printf("%-36s n=%d  pinball X %.5f ± %.5f (quiet %.5f, active %.5f)  P %.5f  cover90 X %.3f (quiet %.3f, active %.3f)  cover50 X %.3f  width90 X %.3f (quiet %.3f, active %.3f)  cover90 P %.3f  BSS M %.3f ± %.3f  BSS M new %.3f  BSS S1 %.3f\n",
            r.model, r.n_seeds, r.pinball_x, r.pinball_x_sd, r.pinball_x_quiet, r.pinball_x_active, r.pinball_p,
            r.cover90_x, r.cover90_x_quiet, r.cover90_x_active, r.cover50_x, r.width90_x, r.width90_x_quiet,
            r.width90_x_active, r.cover90_p, r.bss_m, r.bss_m_sd, r.bss_m_new, r.bss_s1)
end

# reliability tables + diagram for "≥M1 within 60 min" and "≥S1 within 60 min" (seed 1)
order = ["Persistence + static error bars", "Best model + static error bars", "Best model + learned quantile head"]
colors = Dict(order[1] => MODEL_COLORS["Persistence"], order[2] => "#2a78d6", order[3] => MODEL_COLORS["Physics, per-channel λ"])
rel = DataFrame()
for name in order, (ev, pk, ok) in (("≥M1", :prob_m, :obs_m), ("≥S1", :prob_s1, :obs_s1))
    rt = reliability(getfield(probs[name], pk), getfield(probs[name], ok))
    rt.model .= name; rt.event .= ev
    append!(rel, rt)
end
open(io -> show(io, MIME("text/csv"), rel), joinpath(RESULTS_DIR, "prob_reliability.csv"), "w")

fig = Figure(size = (1000, 440))
for (j, ev) in enumerate(("≥M1", "≥S1"))
    ax = Axis(fig[1, j]; title = "Reliability: $(ev) within the next 60 min (test, seed 1)",
              xlabel = "Forecast probability", ylabel = "Observed frequency", limits = (0, 1, 0, 1), aspect = 1)
    lines!(ax, [0, 1], [0, 1]; color = (:black, 0.3), linestyle = :dash)
    for name in order
        r = filter(x -> x.model == name && x.event == ev && x.n >= 20, rel)
        nrow(r) == 0 && continue
        scatterlines!(ax, r.p_mean, r.o_freq; color = colors[name], label = name, linewidth = 2, markersize = 9)
    end
    j == 1 && axislegend(ax; position = :lt, framevisible = false)
end
Label(fig[2, 1:2], "Bins with fewer than 20 forecasts are omitted. On the diagonal = calibrated.";
      fontsize = 12, color = :gray40)
save(joinpath(FIG_DIR, "prob_reliability.png"), fig; px_per_unit = 2)

# fan-chart examples: learned 50% / 90% intervals vs static ones, X-ray channel, seed 1
fig = Figure(size = (1200, 420))
W, H = data.W, data.H
lg(v) = v .* s.σ[1] .+ s.μ[1]
ax1 = Ref{Any}(nothing)
for (j, (title, n)) in enumerate(cases)
    e = examples[n]
    ax = Axis(fig[1, j]; title, xlabel = "Minutes from forecast time", ylabel = j == 1 ? "log₁₀ XRS-B (W m⁻²)" : "")
    th, tf = -119:0, 1:H
    band!(ax, tf, lg(e.static[:, 1]), lg(e.static[:, 5]); color = (colors[order[2]], 0.12))
    lines!(ax, tf, lg(e.static[:, 1]); color = colors[order[2]], linestyle = :dash, linewidth = 1, label = "Static 90% interval")
    lines!(ax, tf, lg(e.static[:, 5]); color = colors[order[2]], linestyle = :dash, linewidth = 1)
    band!(ax, tf, lg(e.head[:, 1]), lg(e.head[:, 5]); color = (colors[order[3]], 0.18), label = "Learned 90% interval")
    band!(ax, tf, lg(e.head[:, 2]), lg(e.head[:, 4]); color = (colors[order[3]], 0.35), label = "Learned 50% interval")
    lines!(ax, tf, lg(e.point); color = colors[order[3]], linewidth = 2, label = "Point forecast")
    lines!(ax, th, lg(data.test.X[W-119:W, 1, n]); color = :gray30, linewidth = 1, label = "Observed")
    lines!(ax, tf, lg(data.test.Y[:, 1, n]); color = :black, linewidth = 1.5, label = "Observed (future)")
    vlines!(ax, 0; color = (:black, 0.25))
    j == 1 && (ax1[] = ax)
end
Legend(fig[2, 1:3], ax1[]; orientation = :horizontal, framevisible = false, labelsize = 11)
save(joinpath(FIG_DIR, "prob_examples.png"), fig; px_per_unit = 2)
println("Figures → figures/prob_reliability.png, figures/prob_examples.png")
