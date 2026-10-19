# Multi-year real data from the NCEI GOES-R archive (science-quality L2, 1-minute averages).
#
#   XRS : yearly files  sci_xrsf-l2-avg1m_g<NN>_y<YYYY>_v2-2-2.nc   (~33 MB/yr)  → xrsb_flux + xrsb_flag
#   SGPS: daily files   sci_sgps-l2-avg1m_g<NN>_d<YYYYMMDD>_v*.nc    (~2.3 MB/day) → 13 differential proton
#         channels (1–500 MeV, two sensor heads) + a >500 MeV integral channel.
#
# SGPS has no ≥10 MeV integral channel (SWPC builds theirs operationally), so `integral_above` reconstructs it as a
# bin sum of the differential channels above 10 MeV plus the >500 MeV integral channel. The two heads are averaged,
# since yaw flips swap which one looks east/west. `validate_vs_swpc.jl` compares the result with SWPC's product.

const NCEI_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
const NCEI_DIR = joinpath(pkgdir(@__MODULE__), "data", "ncei")
const XRS_FLAG_KEEP = (0x00, 0x04)        # good_data, interpolated; drop eclipse (bit 1) and bad_data (bit 2)

ncei_xrs_url(sat::Int, year::Int) =
    "$NCEI_BASE/goes$sat/l2/data/xrsf-l2-avg1m_science/sci_xrsf-l2-avg1m_g$(sat)_y$(year)_v2-2-2.nc"
ncei_sgps_dir(sat::Int, d::Date) =
    "$NCEI_BASE/goes$sat/l2/data/sgps-l2-avg1m/$(Dates.format(d, "yyyy"))/$(Dates.format(d, "mm"))/"

"Download `url` to `path` unless it is already cached. Returns the path (or `nothing` on HTTP failure)."
function fetch_cached(url::AbstractString, path::AbstractString)
    isfile(path) && filesize(path) > 0 && return path
    mkpath(dirname(path))
    try
        Downloads.download(url, path * ".part")
        mv(path * ".part", path; force = true)
        return path
    catch err
        rm(path * ".part"; force = true)
        @warn "download failed" url err
        return nothing
    end
end

"Find the daily SGPS file name for `d` (the version suffix varies) from the month's directory listing."
function sgps_filename(sat::Int, d::Date; listing_cache = Dict{String,String}())
    dir = ncei_sgps_dir(sat, d)
    html = get!(listing_cache, dir) do
        try
            String(take!(Downloads.download(dir, IOBuffer())))
        catch
            ""
        end
    end
    m = match(Regex("sci_sgps-l2-avg1m_g$(sat)_d$(Dates.format(d, "yyyymmdd"))_v[0-9-]+\\.nc"), html)
    return m === nothing ? nothing : m.match
end

"XRS-B 1-minute flux with quality filtering: DataFrame(time, xray, flag)."
function read_xrs_nc(path::AbstractString; keep = XRS_FLAG_KEEP)
    NCDataset(path) do ds
        t = ds["time"][:]
        f = ds["xrsb_flux"][:]
        q = ds["xrsb_flag"].var[:]
        ok = [!ismissing(t[i]) && !ismissing(f[i]) && f[i] > 0 && q[i] in keep for i in eachindex(t)]
        DataFrame(time = DateTime.(t[ok]), xray = Float64.(f[ok]), flag = q[ok])
    end
end

"""
    integral_above(j, El, Eu, jint; Emin = 1e4)

Integral flux (pfu) above `Emin` keV from differential fluxes `j` [cm⁻² sr⁻¹ keV⁻¹ s⁻¹] in channels with edges
`El`, `Eu` [keV]: Σ j·ΔE over the part of each channel above `Emin`, plus the >500 MeV integral channel `jint`.
A plain bin sum, deliberately: at quiet times the 1-min channels are count-limited (many exact zeros), so
spectral fits through them are unstable. Inter-channel gaps are not filled (an underestimate during events, not yet
quantified), and overlapping channel edges are clipped so no energy is counted twice.
"""
function integral_above(j::AbstractVector, El::AbstractVector, Eu::AbstractVector, jint; Emin = 1e4)
    order = sortperm(collect(El))
    total, covered = 0.0, Float64(Emin)
    for i in order
        (ismissing(j[i]) || j[i] < 0) && continue
        lo = max(Float64(El[i]), covered)
        hi = Float64(Eu[i])
        hi > lo || continue
        total += Float64(j[i]) * (hi - lo)
        covered = hi
    end
    return total + (ismissing(jint) || jint < 0 ? 0.0 : Float64(jint))
end

"≥10 MeV integral protons (pfu) from one daily SGPS file, mean of the two heads: DataFrame(time, proton)."
function read_sgps_nc(path::AbstractString; Emin = 1e4)
    NCDataset(path) do ds
        t = ds["time"][:]
        J = ds["AvgDiffProtonFlux"][:, :, :]            # (13, 2, T)
        Ji = ds["AvgIntProtonFlux"][:, :]               # (2, T)
        El = ds["DiffProtonLowerEnergy"][:, :]
        Eu = ds["DiffProtonUpperEnergy"][:, :]
        out = Vector{Union{Missing,Float64}}(missing, length(t))
        for n in eachindex(t)
            v = [integral_above(view(J, :, h, n), view(El, :, h), view(Eu, :, h), Ji[h, n]; Emin) for h in 1:2]
            v = collect(skipmissing(v))
            isempty(v) || (out[n] = sum(v) / length(v))
        end
        ok = .!ismissing.(t) .& .!ismissing.(out)
        DataFrame(time = DateTime.(t[ok]), proton = Float64.(out[ok]))
    end
end

"""
    load_ncei(sat, d0, d1; dir = NCEI_DIR, max_gap = 10) -> DataFrame

Download (cached under `data/ncei/`) and merge GOES-`sat` XRS-B and reconstructed ≥10 MeV protons for the dates
d0…d1, quality-filtered, on a 1-minute grid in the pipeline schema (`time, minute, xray, proton`).
"""
function load_ncei(sat::Int, d0::Date, d1::Date; dir = NCEI_DIR, max_gap = 10, verbose = true)
    xs = DataFrame[]
    for y in year(d0):year(d1)
        f = fetch_cached(ncei_xrs_url(sat, y), joinpath(dir, "xrs", "sci_xrsf-l2-avg1m_g$(sat)_y$(y)_v2-2-2.nc"))
        f === nothing || push!(xs, read_xrs_nc(f))
    end
    x = filter(r -> d0 <= Date(r.time) <= d1, vcat(xs...))
    listing = Dict{String,String}()
    ps = DataFrame[]
    for (k, d) in enumerate(d0:Day(1):d1)
        name = sgps_filename(sat, d; listing_cache = listing)
        name === nothing && continue
        f = fetch_cached(ncei_sgps_dir(sat, d) * name, joinpath(dir, "sgps", "g$(sat)", name))
        f === nothing || push!(ps, read_sgps_nc(f))
        verbose && k % 30 == 0 && println("  SGPS through ", d)
    end
    p = vcat(ps...)
    rename!(x, :xray => :flux); rename!(p, :proton => :flux)
    return swpc_frame(select(x, :time, :flux), p; max_gap)
end
