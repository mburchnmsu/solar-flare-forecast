# Experiment G1 scoring: persistence, R4 (trained on GOES-16 only) and G1 (GOES-16 + GOES-18) on three test sets:
#   GOES-16 test 2024-07 → 2025-03 (the R-series protocol), GOES-18 over the same period (another instrument), and
#   GOES-18 2025-04 → 2026-10 (out-of-time: after all training/selection data). Scored against observations, dex.
#     julia -t 1 --project=. sweeps/analyze_g18.jl
# Output: results/g18_metrics.csv, results/g18_summary.csv, results/g18_crosscal.csv
include(joinpath(@__DIR__, "g18.jl"))

r4 = "real_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr"
g1 = "g1618_physp_1e4_0.01_ft-physp_1e4_0.01_s1-pr"
load(n) = (f = joinpath(@__DIR__, "results", n, "run.jls"); isfile(f) ? deserialize(f) : nothing)
R4 = filter(!isnothing, [load("$(r4)_s$k") for k in 1:3])
G1 = filter(!isnothing, [load("$(g1)_s$k") for k in 1:3])
scaler = R4[1].scaler
df16, d16 = real_data(; scaler)
df18, d18 = g18_windows(scaler)
println("R4 seeds ", length(R4), ", G1 seeds ", length(G1), "; test windows: G16 ", length(d16.test), ", G18 ",
        length(d18.test), ", G18 new ", length(d18.new))

# instrument cross-check: GOES-18 vs GOES-16 over the minutes both have (log ratio GOES-18 / GOES-16, dex)
j = innerjoin(select(df16, :time, :xray => :x16, :proton => :p16), select(df18, :time, :xray => :x18, :proton => :p18); on = :time)
iqrange(v) = quantile(v, 0.75) - quantile(v, 0.25)
lx, lp = log10.(j.x18) .- log10.(j.x16), log10.(j.p18) .- log10.(j.p16)
subsets = [("XRS-B, all", lx, trues(nrow(j))), ("XRS-B, ≥M1 (GOES-16)", lx, j.x16 .>= 1e-5),
           ("Protons, all", lp, trues(nrow(j))), ("Protons, ≥10 pfu (GOES-16)", lp, j.p16 .>= 10)]
cc = DataFrame([(; subset = n, minutes = count(m), median_dex = median(v[m]), iqr_dex = iqrange(v[m])) for (n, v, m0) in subsets for m in (m0 .& isfinite.(v),)])
open(io -> show(io, MIME("text/csv"), cc), joinpath(RESULTS_DIR, "g18_crosscal.csv"), "w")
show(stdout, cc; eltypes = false); println()

const KEYS = (:rmse_x, :rmse_x_quiet, :rmse_x_active, :rmse_p, :rough_quiet, :tss_m, :tss_m_new, :tss_s1)
ev(ŷ, w) = evaluate(ŷ, w, scaler; Yclean = w.Y, loss = PhysicsLoss())
sets = [("GOES-16 test 2024-07→2025-03", d16.test), ("GOES-18 test 2024-07→2025-03", d18.test),
        ("GOES-18 new 2025-04→2026-10", d18.new)]
rows = NamedTuple[]
for (set, w) in sets
    e = ev(persistence(w), w)
    push!(rows, (; set, model = "Persistence", seed = 0, (k => getfield(e, k) for k in KEYS)...))
end
for (k, r) in enumerate(R4)
    model = build_model(180, 60); Flux.loadmodel!(model, r.state)
    for (set, w) in sets
        e = ev(predict(model, w.X), w)
        push!(rows, (; set, model = "R4: GOES-16 only", seed = k, (q => getfield(e, q) for q in KEYS)...))
    end
end
for (k, r) in enumerate(G1)
    for ((set, w), p) in zip(sets, (r.pred_test, r.pred_test_g18, r.pred_new_g18))
        e = ev(p, w)
        push!(rows, (; set, model = "G1: GOES-16 + GOES-18", seed = k, (q => getfield(e, q) for q in KEYS)...))
    end
end
t = DataFrame(rows)
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "g18_metrics.csv"), "w")
summary = combine(groupby(t, [:set, :model]; sort = false), nrow => :n_seeds, [k => mean => k for k in KEYS]...,
                  [k => (v -> length(v) > 1 ? std(v) : NaN) => Symbol(k, :_sd) for k in KEYS]...)
open(io -> show(io, MIME("text/csv"), summary), joinpath(RESULTS_DIR, "g18_summary.csv"), "w")
for r in eachrow(summary)
    @printf("%-30s %-24s n=%d  X-ray %.4f ± %.4f  active %.4f  protons %.4f ± %.4f  TSS≥M %.3f  TSS≥M new %.3f  TSS S1 %.3f\n",
            r.set, r.model, r.n_seeds, r.rmse_x, r.rmse_x_sd, r.rmse_x_active, r.rmse_p, r.rmse_p_sd, r.tss_m,
            r.tss_m_new, r.tss_s1)
end
