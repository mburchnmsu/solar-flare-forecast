# Recalibrate the P1 event probabilities ("≥M1 / ≥S1 within 60 min") with isotonic regression fit on validation.
#     julia -t 1 --project=. sweeps/recalibrate_real.jl [base_config]
# Needs the P1 runs (sweeps/results/prob_<base>_s<k>/run.jls). No training: the quantile heads and static error bars
# are reloaded, validation probabilities fit the isotonic map, test is scored once before and after.
# Output: results/recal_metrics.csv, results/recal_summary.csv, results/recal_reliability.csv, figures/recal_reliability.png
include(joinpath(@__DIR__, "train_real.jl"))           # real_data(), sol_setup
using Serialization, Flux

base = length(ARGS) > 0 ? ARGS[1] : "real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr"
runs = [deserialize(joinpath(@__DIR__, "results", "$(base)_s$(k)", "run.jls")) for k in 1:3]
probruns = [deserialize(joinpath(@__DIR__, "results", "prob_$(base)_s$(k)", "run.jls")) for k in 1:3]
df, data = real_data(; scaler = runs[1].scaler)
s, σx = data.scaler, data.scaler.σ[1]
τ = QUANTILE_LEVELS

logy(Y, c) = Y[:, c, :] .* s.σ[c] .+ s.μ[c]
clim_m = mean(vec(maximum(logy(data.val.Y, 1); dims = 1)) .>= -5)
clim_s1 = mean(vec(maximum(logy(data.val.Y, 2); dims = 1)) .>= 1)
new_test = vec(data.test.X[end, 1, :]) .* s.σ[1] .+ s.μ[1] .< -5

events = (("≥M1", :prob_m, :obs_m, clim_m), ("≥S1", :prob_s1, :obs_s1, clim_s1))
rows = NamedTuple[]
rel = DataFrame()
function add!(name, k, scv, sct)
    r = Dict{Symbol,Any}(:model => name, :seed => k)
    for (ev, pk, ok, clim) in events
        tag = ev == "≥M1" ? "m" : "s1"
        pv, ov, pt, ot = getfield(scv, pk), getfield(scv, ok), getfield(sct, pk), getfield(sct, ok)
        cal = IsotonicCalibration(pv, ov)
        pc = cal(pt)
        raw, rec = brier(pt, ot, clim), brier(pc, ot, clim)
        r[Symbol("bss_$(tag)_raw")] = raw.bss
        r[Symbol("bss_$(tag)_cal")] = rec.bss
        r[Symbol("mean_p_$(tag)_raw")] = mean(pt)
        r[Symbol("mean_p_$(tag)_cal")] = mean(pc)
        r[Symbol("obs_$(tag)")] = mean(ot)
        if tag == "m"
            r[:bss_m_new_raw] = brier(pt[new_test], ot[new_test], clim).bss
            r[:bss_m_new_cal] = brier(pc[new_test], ot[new_test], clim).bss
        end
        if k <= 1
            for (stage, p) in (("raw", pt), ("recalibrated", pc))
                rt = reliability(p, ot); rt.model .= name; rt.event .= ev; rt.stage .= stage
                append!(rel, rt)
            end
        end
    end
    push!(rows, (; (k => r[k] for k in sort!(collect(keys(r))))...))
end

# persistence + static error bars (the same for every seed)
emp_pers = probruns[1].emp_persistence
pv, pt = persistence(data.val), persistence(data.test)
add!("Persistence + static error bars", 0,
     score_quantiles(apply_quantiles(emp_pers, pv, data.val.X, σx), data.val, s; clim_m, clim_s1),
     score_quantiles(apply_quantiles(emp_pers, pt, data.test.X, σx), data.test, s; clim_m, clim_s1))

for (k, (r, pr)) in enumerate(zip(runs, probruns))
    model = build_model(data.W, data.H)
    Flux.loadmodel!(model, r.state)
    Fva, Pva = trunk_features(model, data.val.X)
    Fte, Pte = trunk_features(model, data.test.X)
    add!("Best model + static error bars", k,
         score_quantiles(apply_quantiles(pr.emp_point, Pva, data.val.X, σx), data.val, s; clim_m, clim_s1),
         score_quantiles(apply_quantiles(pr.emp_point, Pte, data.test.X, σx), data.test, s; clim_m, clim_s1))
    head = build_quantile_head(data.H; τ = pr.τ)
    Flux.loadmodel!(head, pr.head)
    add!("Best model + learned quantile head", k,
         score_quantiles(predict_quantiles(head, Fva, Pva), data.val, s; clim_m, clim_s1),
         score_quantiles(predict_quantiles(head, Fte, Pte), data.test, s; clim_m, clim_s1))
    println("seed $k done"); flush(stdout)
end

t = select(DataFrame(rows), :model, :seed, :)
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "recal_metrics.csv"), "w")
keys_ = filter(k -> !(k in (:model, :seed)), propertynames(t))
summary = combine(groupby(t, :model, sort = false), nrow => :n_seeds, [k => mean => k for k in keys_]...,
                  [k => (v -> length(v) > 1 ? std(v) : NaN) => Symbol(k, :_sd) for k in keys_]...)
open(io -> show(io, MIME("text/csv"), summary), joinpath(RESULTS_DIR, "recal_summary.csv"), "w")
open(io -> show(io, MIME("text/csv"), rel), joinpath(RESULTS_DIR, "recal_reliability.csv"), "w")
for r in eachrow(summary)
    @printf("%-36s BSS M %.3f → %.3f (±%.3f)  BSS M new %.3f → %.3f  BSS S1 %.3f → %.3f (±%.3f)  mean p M %.4f → %.4f (obs %.4f)  S1 %.4f → %.4f (obs %.4f)\n",
            r.model, r.bss_m_raw, r.bss_m_cal, r.bss_m_cal_sd, r.bss_m_new_raw, r.bss_m_new_cal, r.bss_s1_raw,
            r.bss_s1_cal, r.bss_s1_cal_sd, r.mean_p_m_raw, r.mean_p_m_cal, r.obs_m, r.mean_p_s1_raw, r.mean_p_s1_cal, r.obs_s1)
end

# reliability before / after for the learned head (seed 1)
name = "Best model + learned quantile head"
fig = Figure(size = (1000, 440))
for (j, ev) in enumerate(("≥M1", "≥S1"))
    ax = Axis(fig[1, j]; title = "$(ev) within the next 60 min (test, seed 1)",
              xlabel = "Forecast probability", ylabel = "Observed frequency", limits = (0, 1, 0, 1), aspect = 1)
    lines!(ax, [0, 1], [0, 1]; color = (:black, 0.3), linestyle = :dash)
    for (stage, col) in (("raw", "#9aa4b1"), ("recalibrated", MODEL_COLORS["Physics, per-channel λ"]))
        r = filter(x -> x.model == name && x.event == ev && x.stage == stage && x.n >= 20, rel)
        nrow(r) == 0 && continue
        scatterlines!(ax, r.p_mean, r.o_freq; color = col, linewidth = 2, markersize = 9,
                      label = stage == "raw" ? "Quantile head, raw" : "After isotonic recalibration (fit on validation)")
    end
    j == 1 && axislegend(ax; position = :lt, framevisible = false)
end
Label(fig[2, 1:2], "Bins with fewer than 20 forecasts are omitted. On the diagonal = calibrated.";
      fontsize = 12, color = :gray40)
save(joinpath(FIG_DIR, "recal_reliability.png"), fig; px_per_unit = 2)
println("Figure → figures/recal_reliability.png")
