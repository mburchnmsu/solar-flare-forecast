# Real-data ingest from NOAA SWPC. Every loader returns the same schema as `generate_telemetry`
# (`time`, `minute`, `xray` [W m^-2, XRS-B 0.1-0.8 nm], `proton` [pfu, >=10 MeV]) so the rest of the
# pipeline does not care where the data came from.
#
# SWPC JSON products roll over (the 7-day file only ever holds the last 7 days), so every download is
# cached to data/raw/ with a UTC timestamp; merging the cache builds a longer history over time.
# For multi-year training sets use the NCEI GOES-R archive instead (XRS `xrsf-l2-avg1m_science` and
# SGPS netCDF files; read with NCDatasets.jl and write the same three columns).

const SWPC_BASE = "https://services.swpc.noaa.gov/json/goes"
const SWPC_SPANS = ("6-hour", "1-day", "3-day", "7-day")

swpc_url(product::AbstractString, span::AbstractString; satellite = "primary") =
    "$SWPC_BASE/$satellite/$product-$span.json"

"Download a SWPC JSON product, cache the raw bytes under `cache_dir`, and parse it."
function fetch_swpc_json(product::AbstractString; span = "7-day", satellite = "primary",
                         cache_dir = joinpath(pkgdir(@__MODULE__), "data", "raw"))
    span in SWPC_SPANS || throw(ArgumentError("span must be one of $SWPC_SPANS"))
    mkpath(cache_dir)
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    path = joinpath(cache_dir, "$(product)_$(span)_$(satellite)_$(stamp).json")
    Downloads.download(swpc_url(product, span; satellite), path)
    return JSON3.read(read(path, String)), path
end

parse_swpc_time(s::AbstractString) = DateTime(first(s, 19), dateformat"yyyy-mm-ddTHH:MM:SS")

"Rows of a SWPC product for one `energy` label as (time, flux) columns; drops fill/non-positive values."
function swpc_channel(records, energy::AbstractString)
    t = DateTime[]
    f = Float64[]
    for r in records
        String(r.energy) == energy || continue
        v = r.flux
        (v === nothing || !(v isa Real) || v <= 0) && continue
        push!(t, parse_swpc_time(String(r.time_tag)))
        push!(f, Float64(v))
    end
    return DataFrame(time = t, flux = f)
end

"""
    regrid_1min(df, col; max_gap = 10)

Put an irregular (time, value) series on a strict 1-minute grid. Gaps up to `max_gap` minutes are filled by
linear interpolation in log10 flux (the natural space for GOES data); longer gaps stay `missing`.
"""
function regrid_1min(times::Vector{DateTime}, vals::Vector{Float64}, grid::Vector{DateTime}; max_gap = 10)
    out = Vector{Union{Missing,Float64}}(missing, length(grid))
    isempty(times) && return out
    p = sortperm(times)
    times, lv = times[p], log10.(vals[p])
    j = 1
    for (i, g) in enumerate(grid)
        while j < length(times) && times[j+1] <= g
            j += 1
        end
        if times[j] == g
            out[i] = 10^lv[j]
        elseif j < length(times) && times[j] < g < times[j+1]
            gap = Dates.value(times[j+1] - times[j]) / 60_000
            gap <= max_gap || continue
            w = Dates.value(g - times[j]) / Dates.value(times[j+1] - times[j])
            out[i] = 10^((1 - w) * lv[j] + w * lv[j+1])
        end
    end
    return out
end

"""
    load_swpc(; span = "7-day", satellite = "primary", max_gap = 10) -> DataFrame

Pull live GOES XRS-B and >=10 MeV integral protons from SWPC and return them on a common 1-minute grid in the
pipeline schema. Rows with a gap longer than `max_gap` minutes in either channel are dropped; `make_windows`
then never builds a window across a dropped row because it checks for contiguous minutes.
"""
function load_swpc(; span = "7-day", satellite = "primary", max_gap = 10)
    xr, _ = fetch_swpc_json("xrays"; span, satellite)
    pr, _ = fetch_swpc_json("integral-protons"; span, satellite)
    return swpc_frame(swpc_channel(xr, "0.1-0.8nm"), swpc_channel(pr, ">=10 MeV"); max_gap)
end

"""
    load_swpc_cache(; cache_dir, satellite = "primary", max_gap = 10) -> DataFrame

Rebuild a (possibly much longer than 7-day) series from every cached SWPC download, without touching the network.
Overlapping downloads are merged and de-duplicated by timestamp, so running `load_swpc()` once a week and
then calling this grows a continuous training set.
"""
function load_swpc_cache(; cache_dir = joinpath(pkgdir(@__MODULE__), "data", "raw"), satellite = "primary", max_gap = 10)
    files = readdir(cache_dir; join = true)
    pick(prefix) = filter(f -> startswith(basename(f), prefix) && occursin("_$(satellite)_", f), files)
    merge(fs, energy) = unique!(vcat((swpc_channel(JSON3.read(read(f, String)), energy) for f in fs)...), :time)
    x = merge(pick("xrays_"), "0.1-0.8nm")
    p = merge(pick("integral-protons_"), ">=10 MeV")
    (nrow(x) == 0 || nrow(p) == 0) && error("no cached SWPC files for satellite '$satellite' in $cache_dir")
    return swpc_frame(x, p; max_gap)
end

"Put the X-ray and proton channels on one 1-minute grid (pipeline schema), dropping rows inside long gaps."
function swpc_frame(x::DataFrame, p::DataFrame; max_gap = 10)
    t0 = max(minimum(x.time), minimum(p.time))
    t1 = min(maximum(x.time), maximum(p.time))
    grid = collect(t0:Minute(1):t1)
    df = DataFrame(time = grid,
                   xray = regrid_1min(x.time, x.flux, grid; max_gap),
                   proton = regrid_1min(p.time, p.flux, grid; max_gap))
    df.minute = Dates.value.(df.time .- t0) .÷ 60_000
    dropmissing!(df)
    df.xray = Float64.(df.xray)
    df.proton = Float64.(df.proton)
    return select!(df, :time, :minute, :xray, :proton)
end

"""
    read_telemetry_csv(path) -> DataFrame

Load a CSV with columns `time`, `xray`, `proton` (e.g. an NCEI extract you saved yourself) in the pipeline schema.
"""
function read_telemetry_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    df.time = DateTime.(df.time)
    df.minute = Dates.value.(df.time .- first(df.time)) .÷ 60_000
    return select!(df, :time, :minute, :xray, :proton)
end
