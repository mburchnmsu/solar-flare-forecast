# Validate the reconstructed ≥10 MeV integral on real SEP events: NOAA SWPC's official 2024 event list
# (ftp.swpc.noaa.gov/pub/indices/SPE.txt) vs the peak of our GOES-16 SGPS reconstruction around each listed maximum.
#     julia -t 1 --project=. sweeps/validate_sep_peaks.jl
include(joinpath(@__DIR__, "..", "sol_setup.jl"))

# (listed maximum time UT, listed peak pfu) — 2024 entries of SPE.txt
const SPE_2024 = [
    (DateTime(2024, 1, 4, 8, 35), 20.0), (DateTime(2024, 1, 29, 18, 5), 137.0), (DateTime(2024, 2, 9, 23, 55), 187.0),
    (DateTime(2024, 5, 10, 17, 45), 207.0), (DateTime(2024, 5, 14, 5, 20), 121.0), (DateTime(2024, 9, 9, 22, 50), 34.0),
    (DateTime(2024, 9, 17, 10, 50), 33.0), (DateTime(2024, 10, 10, 15, 15), 1810.0),
]   # (the Nov 21 entry lists a maximum before its start, so it is left out)

listing = Dict{String,String}()
rows = NamedTuple[]
for (tmax, peak) in SPE_2024
    days = Date(tmax - Hour(12)):Day(1):Date(tmax + Hour(12))
    ps = DataFrame[]
    for d in days
        name = SolarFlareForecast.sgps_filename(16, d; listing_cache = listing)
        name === nothing && continue
        f = SolarFlareForecast.fetch_cached(SolarFlareForecast.ncei_sgps_dir(16, d) * name,
                                            joinpath(SolarFlareForecast.NCEI_DIR, "sgps", "g16", name))
        f === nothing || push!(ps, read_sgps_nc(f))
    end
    isempty(ps) && continue
    p = filter(r -> abs(r.time - tmax) <= Hour(12), vcat(ps...))
    # 5-min running median, so a single noisy minute can't set the peak (SWPC's product is 5-min based)
    v = [median(p.proton[max(1, i - 2):min(end, i + 2)]) for i in 1:nrow(p)]
    k = argmax(v)
    push!(rows, (; listed_max = tmax, listed_pfu = peak, recon_pfu = round(v[k]; sigdigits = 3),
                 ratio = round(v[k] / peak; digits = 2), recon_max = p.time[k]))
end
t = DataFrame(rows)
show(stdout, t; allrows = true, eltypes = false); println()
@printf("median ratio %.2f, log10 ratio scatter (std) %.2f dex, n = %d\n", median(t.ratio), std(log10.(t.ratio)), nrow(t))
open(io -> show(io, MIME("text/csv"), t), joinpath(RESULTS_DIR, "sep_peak_validation.csv"), "w")
