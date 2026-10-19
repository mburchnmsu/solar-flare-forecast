# Download + merge GOES-16 real data (NCEI L2, quality-filtered) for real-data training; cached.
#     julia -t 1 --project=. sweeps/fetch_ncei.jl [sat] [yyyy-mm-dd start] [yyyy-mm-dd end]
include(joinpath(@__DIR__, "..", "sol_setup.jl"))
using Serialization
sat = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 16
d0 = length(ARGS) > 1 ? Date(ARGS[2]) : Date(2022, 1, 1)
d1 = length(ARGS) > 2 ? Date(ARGS[3]) : Date(2025, 3, 31)
out = joinpath(SolarFlareForecast.NCEI_DIR, "merged_g$(sat)_$(d0)_$(d1).jls")
t = @elapsed df = load_ncei(sat, d0, d1)
serialize(out, df)
expected = Dates.value(DateTime(d1) + Day(1) - DateTime(d0)) ÷ 60_000
@printf("%d rows (%.1f%% of %d minutes) in %.0f s → %s\n", nrow(df), 100nrow(df) / expected, expected, t, out)
@printf("XRS-B range %.2e–%.2e W/m², max class %s; protons ≥10 pfu %.2f%% of minutes, max %.0f pfu\n",
        extrema(df.xray)..., flare_class(maximum(df.xray)), 100mean(df.proton .>= 10), maximum(df.proton))
