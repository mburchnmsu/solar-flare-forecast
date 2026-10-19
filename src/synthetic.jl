# Synthetic GOES-like telemetry: 1-minute XRS-B (0.1-0.8 nm) flux and >=10 MeV integral proton flux.
#
# Physics baked into the generator (each one is something a real forecaster would recognise):
#   * Background X-ray level follows a slow "activity" process (27-day rotational modulation + OU noise).
#   * Flare occurrence is a doubly-stochastic, self-exciting (Hawkes) Poisson process: the rate rises with
#     activity and briefly after each flare (sympathetic flaring), so the recent past carries real information.
#   * Peak fluxes follow the observed power law dN/dF ∝ F^-α with α ≈ 1.9.
#   * Light curves: half-Gaussian impulsive rise, exponential (thermal/conductive cooling) decay, with
#     longer timescales for bigger flares. Flares add in *linear* flux on top of the background.
#   * Large flares can launch SEP events: delayed onset (particle transport), hours-long rise, ~day decay.
# Both noise-free ("clean") and instrument-noise versions are returned so models can be scored against truth.

const XRAY_CLASS_FLOOR = (A = 1e-8, B = 1e-7, C = 1e-6, M = 1e-5, X = 1e-4)   # W m^-2
const SEP_THRESHOLD_PFU = 10.0                                              # NOAA S1 threshold

Base.@kwdef struct TelemetryConfig
    n_days::Int = 90
    cadence_min::Int = 1
    bg_log10_mean::Float64 = -6.6        # quiet-Sun-to-moderate XRS-B background (≈ B2.5)
    bg_rotation_amp::Float64 = 0.30      # dex, 27-day modulation
    bg_ou_sigma::Float64 = 0.20          # dex
    bg_ou_tau_min::Float64 = 8 * 60.0
    bg_smooth_min::Float64 = 45.0        # low-pass timescale of the background
    flare_rate_per_day::Float64 = 6.0    # base rate of flares above flare_fmin
    flare_fmin::Float64 = 3e-7           # W m^-2
    flare_fmax::Float64 = 2e-3
    flare_alpha::Float64 = 1.9
    activity_coupling::Float64 = 1.5     # rate ∝ exp(coupling * activity anomaly in dex)
    hawkes_boost::Float64 = 0.3          # extra rate (× base) right after a flare...
    hawkes_tau_min::Float64 = 90.0       # ...decaying on this timescale
    xray_noise_frac::Float64 = 0.02      # multiplicative 1σ
    proton_bg_pfu::Float64 = 0.25
    proton_noise_frac::Float64 = 0.06
    seed::Int = 42
end

"Peak flux drawn from a truncated power law dN/dF ∝ F^-α via inverse-CDF sampling."
function sample_peak_flux(rng::AbstractRNG, fmin::Float64, fmax::Float64, α::Float64)
    a = 1 - α
    u = rand(rng)
    return (fmin^a + u * (fmax^a - fmin^a))^(1 / a)
end

"Forward-backward first-order exponential smoother (zero phase lag), in place."
function smooth_zero_phase!(v::Vector{Float64}, a::Float64)
    @inbounds for i in 2:length(v)
        v[i] = a * v[i-1] + (1 - a) * v[i]
    end
    @inbounds for i in length(v)-1:-1:1
        v[i] = a * v[i+1] + (1 - a) * v[i]
    end
    return v
end

"Add one flare light curve (linear flux) into `buf`, peaking at index `ipk`."
function add_flare!(buf::Vector{Float64}, ipk::Int, fpk::Float64, τr::Float64, τd::Float64)
    n = length(buf)
    i0 = max(1, ipk - ceil(Int, 4τr))
    i1 = min(n, ipk + ceil(Int, 12τd))
    @inbounds for i in i0:i1
        t = Float64(i - ipk)
        buf[i] += t < 0 ? fpk * exp(-0.5 * (t / τr)^2) : fpk * exp(-t / τd)
    end
    return buf
end

"Add one SEP event (linear pfu) into `buf`: smooth rise to `ppk` at `ipk`, exponential decay after."
function add_sep!(buf::Vector{Float64}, ionset::Int, ipk::Int, ppk::Float64, τd::Float64)
    n = length(buf)
    trise = max(ipk - ionset, 1)
    i1 = min(n, ipk + ceil(Int, 10τd))
    @inbounds for i in max(1, ionset):i1
        if i <= ipk
            s = (i - ionset) / trise                   # 0 → 1
            buf[i] += ppk * s^2 * exp(2 * (1 - s))     # smooth, zero-slope onset, peaks at s = 1
        else
            buf[i] += ppk * exp(-(i - ipk) / τd)
        end
    end
    return buf
end

"""
    generate_telemetry(cfg = TelemetryConfig()) -> (df, events)

`df` columns: `time` (DateTime), `minute`, `xray`, `proton` (noisy, physical units), `xray_clean`, `proton_clean`.
`events` lists every injected flare (and its SEP, if any) for later event-based scoring.
"""
function generate_telemetry(cfg::TelemetryConfig = TelemetryConfig())
    rng = Xoshiro(cfg.seed)
    n = cfg.n_days * 24 * 60 ÷ cfg.cadence_min
    dt = Float64(cfg.cadence_min)

    # --- background activity (log10 anomaly, dex) -----------------------------------------
    # OU noise is rough at 1-min cadence; real quiet-Sun soft X-rays are smooth on that scale (the short-term
    # variability is the small flares added below), so the OU path is low-passed with a zero-phase filter.
    ou = Vector{Float64}(undef, n)
    ϕ = exp(-dt / cfg.bg_ou_tau_min)
    x = 0.0
    @inbounds for i in 1:n
        x = ϕ * x + sqrt(1 - ϕ^2) * randn(rng)
        ou[i] = x
    end
    smooth_zero_phase!(ou, exp(-dt / cfg.bg_smooth_min))
    ou .*= cfg.bg_ou_sigma / std(ou)
    phase = 2π * rand(rng)
    activity = [cfg.bg_rotation_amp * sin(2π * i * dt / (27 * 1440) + phase) + ou[i] for i in 1:n]
    xray_clean = @. 10.0^(cfg.bg_log10_mean + activity)

    # --- flares: thinned self-exciting Poisson process ---------------------------------------
    base_rate = cfg.flare_rate_per_day / 1440 * dt     # per step
    excite = 0.0
    ϕh = exp(-dt / cfg.hawkes_tau_min)
    proton_clean = fill(cfg.proton_bg_pfu, n)
    events = DataFrame(peak_index = Int[], peak_flux = Float64[], rise_min = Float64[], decay_min = Float64[],
                       sep = Bool[], sep_peak_pfu = Float64[], sep_peak_index = Int[])
    @inbounds for i in 1:n
        excite *= ϕh
        λ = base_rate * exp(cfg.activity_coupling * activity[i]) * (1 + excite)
        rand(rng) < λ || continue
        fpk = sample_peak_flux(rng, cfg.flare_fmin, cfg.flare_fmax, cfg.flare_alpha)
        size = log10(fpk / 1e-6)                                     # 0 at C1
        τr = clamp(4.0 * 10^(0.15size) * exp(0.4randn(rng)), 1.0, 40.0)
        τd = clamp(12.0 * 10^(0.25size) * exp(0.4randn(rng)), 5.0, 240.0)  # cooling time ≥ 5 min
        ipk = i + ceil(Int, 3τr)                                     # onset-to-peak delay
        add_flare!(xray_clean, ipk, fpk, τr, τd)
        excite += cfg.hawkes_boost

        sep, ppk, isep = false, 0.0, 0
        if fpk >= 1e-5                                              # ≥ M1: chance of a SEP event
            p_sep = clamp(0.10 + 0.30 * log10(fpk / 1e-5), 0.0, 0.85)
            if rand(rng) < p_sep
                sep = true
                ppk = 5.0 * (fpk / 1e-5)^1.3 * 10^(0.5randn(rng))     # pfu
                ionset = ipk + round(Int, 15 + 60rand(rng))            # transport delay
                isep = ionset + round(Int, 60 + 300rand(rng))          # 1–6 h rise
                add_sep!(proton_clean, ionset, isep, ppk, 600 + 1200rand(rng))
            end
        end
        push!(events, (ipk, fpk, τr, τd, sep, ppk, isep))
    end

    xray = @. xray_clean * exp(cfg.xray_noise_frac * $randn(rng, n))
    proton = @. proton_clean * exp(cfg.proton_noise_frac * $randn(rng, n))
    t0 = DateTime(2026, 1, 1)
    df = DataFrame(time = t0 .+ Minute.(cfg.cadence_min .* (0:n-1)), minute = collect(0:n-1),
                   xray = xray, proton = proton, xray_clean = xray_clean, proton_clean = proton_clean)
    return df, filter(r -> r.peak_index <= n, events)
end

"GOES letter class string (e.g. \"M2.3\") for a peak XRS-B flux in W m^-2."
function flare_class(f::Real)
    for (letter, floor) in (('X', 1e-4), ('M', 1e-5), ('C', 1e-6), ('B', 1e-7), ('A', 1e-8))
        f >= floor && return string(letter, round(f / floor; digits = 1))
    end
    return "<A1"
end
