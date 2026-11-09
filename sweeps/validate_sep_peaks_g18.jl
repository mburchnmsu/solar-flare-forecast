# Validate the GOES-18 ≥10 MeV reconstruction on the 2025–2026 SEP events (the G1 test periods) against NOAA SWPC.
#     julia -t 1 --project=. sweeps/validate_sep_peaks_g18.jl
# Reference: daily peaks of ≥10 MeV protons at geosynchronous orbit from SWPC's Joint USAF/NOAA Report of
# Solar-Geophysical Activity (ftp.swpc.noaa.gov/pub/warehouse/<year>/RSGA, retrieved 2026-11-08); SPE.txt ends
# 2025-11. One entry per episode with a daily peak ≥ 13 pfu (the event maximum day). SWPC's values come from its primary
# GOES satellite, not necessarily GOES-18.
include(joinpath(@__DIR__, "g18.jl"))

const SEP_2025_26 = [
    (DateTime(2025, 1, 5, 0, 55), 20.0), (DateTime(2025, 2, 25, 2, 40), 37.0), (DateTime(2025, 4, 1, 4, 25), 146.0),
    (DateTime(2025, 6, 1, 9, 15), 666.0), (DateTime(2025, 8, 27, 7, 40), 13.0), (DateTime(2025, 11, 12, 2, 15), 1456.0),
    (DateTime(2026, 1, 19, 19, 15), 36976.0), (DateTime(2026, 5, 26, 1, 10), 23.0), (DateTime(2026, 7, 30, 20, 55), 18.0),
    (DateTime(2026, 9, 5, 16, 45), 17.0),
]

df = deserialize(G18_FILE)
rows = NamedTuple[]
for (tmax, peak) in SEP_2025_26
    p = df[abs.(df.time .- tmax) .<= Hour(12), :]
    nrow(p) == 0 && (println("no GOES-18 data around ", tmax); continue)
    # 5-min running median, so a single noisy minute can't set the peak (SWPC's product is 5-min based)
    v = [median(p.proton[max(1, i - 2):min(end, i + 2)]) for i in 1:nrow(p)]
    k = argmax(v)
    push!(rows, (; listed_max = tmax, listed_pfu = peak, recon_pfu = round(v[k]; sigdigits = 3),
                 ratio = round(v[k] / peak; digits = 2), recon_max = p.time[k], minutes = nrow(p)))
end
t = DataFrame(rows)
show(stdout, t; allrows = true, eltypes = false); println()
@printf("median ratio %.2f, log10 ratio scatter (std) %.2f dex, n = %d\n", median(t.ratio), std(log10.(t.ratio)), nrow(t))
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "sep_peak_validation_g18.csv"), "w")
