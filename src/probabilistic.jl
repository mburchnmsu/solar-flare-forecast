# Probabilistic forecasts: a quantile head on the frozen features of a trained point forecaster, empirical-error
# baselines, and proper scores (pinball loss, interval coverage, Brier skill, reliability).

"Quantile levels predicted by default: a 90% and a 50% central interval plus the median."
const QUANTILE_LEVELS = Float32[0.05, 0.25, 0.5, 0.75, 0.95]

"""
    QuantileHead(net, H, τ)

Maps the 128 trunk features of a [`ResidualForecaster`](@ref) to `length(τ)` quantile trajectories per channel,
anchored on the point forecast: q₁ = point + r₁, qₖ = qₖ₋₁ + softplus(rₖ). The quantiles can never cross.
"""
struct QuantileHead{C}
    net::C
    H::Int
    τ::Vector{Float32}
end
Flux.@layer QuantileHead trainable = (net,)

"Quantiles (H, 2, Q, B) from trunk features `f` (128, B) and the point forecast `point` (H, 2, B)."
function (m::QuantileHead)(f::AbstractMatrix{Float32}, point::AbstractArray{Float32,3})
    Q = length(m.τ)
    r = reshape(m.net(f), m.H, 2, Q, size(f, 2))
    steps = cat(r[:, :, 1:1, :], softplus.(r[:, :, 2:Q, :]); dims = 3)
    return reshape(point, m.H, 2, 1, :) .+ cumsum(steps; dims = 3)
end

"""
    build_quantile_head(H; τ, rng, width, spread)

Last layer starts at zero weights with biases that place the quantiles at Gaussian offsets of `spread`
(standardised units) around the point forecast, so training starts from a sensible, symmetric interval.
"""
function build_quantile_head(H::Int; τ::Vector{Float32} = QUANTILE_LEVELS, rng::AbstractRNG = Xoshiro(1),
                             width::Int = 64, spread::Float32 = 0.2f0)
    Q = length(τ)
    z = Float32[sqrt(2f0) * erfinv_approx(2t - 1) for t in τ] .* spread
    b = zeros(Float32, 2H * Q)
    b[1:2H] .= z[1]
    for k in 2:Q
        b[2H*(k-1)+1:2H*k] .= log(expm1(z[k] - z[k-1]))         # softplus⁻¹ of the gap
    end
    net = Chain(Dense(128 => width, gelu; init = Flux.glorot_uniform(rng)),
                Dense(zeros(Float32, 2H * Q, width), b))
    return QuantileHead(net, H, τ)
end

"Inverse error function (Giles 2012, single precision); avoids a SpecialFunctions dependency."
function erfinv_approx(x::Float32)
    w = -log((1f0 - x) * (1f0 + x))
    if w < 5f0
        w -= 2.5f0
        p = 2.81022636f-08
        for c in (3.43273939f-07, -3.5233877f-06, -4.39150654f-06, 0.00021858087f0, -0.00125372503f0,
                  -0.00417768164f0, 0.246640727f0, 1.50140941f0)
            p = c + p * w
        end
    else
        w = sqrt(w) - 3f0
        p = -0.000200214257f0
        for c in (0.000100950558f0, 0.00134934322f0, -0.00367342844f0, 0.00573950773f0, -0.0076224613f0,
                  0.00943887047f0, 1.00167406f0, 2.83297682f0)
            p = c + p * w
        end
    end
    return p * x
end

"Trunk features (128, N) and point forecasts (H, 2, N) of a trained forecaster, computed in batches."
function trunk_features(model::ResidualForecaster, X::Array{Float32,3}; batchsize::Int = 4096)
    trunk = model.net[1:end-1]
    N = size(X, 3)
    F = Matrix{Float32}(undef, 128, N)
    P = Array{Float32,3}(undef, model.H, 2, N)
    for i in 1:batchsize:N
        j = min(i + batchsize - 1, N)
        x = X[:, :, i:j]
        f = trunk(x)
        F[:, i:j] = f
        P[:, :, i:j] = reshape(model.net[end](f), model.H, 2, :) .+ x[end:end, 1:2, :]
    end
    return F, P
end

"Pinball loss for one channel: quantiles `q` (H, Q, B), targets `y` (H, B)."
function pinball_channel(q, y, τ::Vector{Float32})
    u = reshape(y, size(y, 1), 1, :) .- q
    t = reshape(τ, 1, :, 1)
    return mean(max.(t .* u, (t .- 1f0) .* u))
end

"Mean pinball (quantile) loss of quantiles `q` (H, 2, Q, B) for targets `y` (H, 2, B) at levels `τ`."
function pinball(q, y, τ::Vector{Float32})
    u = reshape(y, size(y, 1), 2, 1, :) .- q
    t = reshape(τ, 1, 1, :, 1)
    return mean(max.(t .* u, (t .- 1f0) .* u))
end

"""
    train_quantile_head(Ftr, Ptr, Ytr, Fva, Pva, Yva; τ, epochs, batchsize, lr, seed, patience)

Fits a [`QuantileHead`](@ref) on frozen trunk features with the pinball loss; early stopping on validation pinball.
"""
function train_quantile_head(Ftr, Ptr, Ytr, Fva, Pva, Yva; τ::Vector{Float32} = QUANTILE_LEVELS, epochs::Int = 40,
                             batchsize::Int = 1024, lr = 1f-3, seed::Int = 1, patience::Int = 5,
                             verbose::Bool = true, label = "quantile")
    rng = Xoshiro(seed)
    head = build_quantile_head(size(Ptr, 1); τ, rng)
    opt = Flux.setup(OptimiserChain(ClipNorm(1f0), Adam(Float32(lr))), head)
    loader = Flux.DataLoader((Ftr, Ptr, Ytr); batchsize, shuffle = true, rng)
    history = DataFrame(epoch = Int[], train_pinball = Float64[], val_pinball = Float64[])
    best, best_state, stall = Inf, Flux.state(head), 0
    for epoch in 1:epochs
        running, nb = 0.0, 0
        for (f, p, y) in loader
            l, g = Flux.withgradient(h -> pinball(h(f, p), y, τ), head)
            Flux.update!(opt, head, g[1])
            running += l; nb += 1
        end
        val = pinball(predict_quantiles(head, Fva, Pva), Yva, τ)
        push!(history, (epoch, running / nb, val))
        verbose && @printf("[%s] epoch %2d  train %.5f  val pinball %.5f\n", label, epoch, running / nb, val)
        verbose && flush(stdout)
        if val < best
            best, best_state, stall = val, deepcopy(Flux.state(head)), 0
        else
            (stall += 1) >= patience && break
        end
    end
    Flux.loadmodel!(head, best_state)
    return head, history
end

"Quantile forecasts (H, 2, Q, N) for a whole split, in batches."
function predict_quantiles(head::QuantileHead, F::Matrix{Float32}, P::Array{Float32,3}; batchsize::Int = 8192)
    N = size(F, 2)
    out = Array{Float32,4}(undef, head.H, 2, length(head.τ), N)
    for i in 1:batchsize:N
        j = min(i + batchsize - 1, N)
        out[:, :, :, i:j] = head(F[:, i:j], P[:, :, i:j])
    end
    return out
end

"""
    EmpiricalQuantiles(point, Y, X, σx; τ)

Static error bars: the `τ` quantiles of the residual Y − point for each lead time, channel and regime (quiet /
active, from the input window). The baseline a learned uncertainty model has to beat.
"""
struct EmpiricalQuantiles
    τ::Vector{Float32}
    offsets::Array{Float32,4}          # (H, 2, Q, 2): last index 1 = quiet, 2 = active
end

function EmpiricalQuantiles(point::Array{Float32,3}, Y::Array{Float32,3}, X::Array{Float32,3}, σx::Float32;
                            τ::Vector{Float32} = QUANTILE_LEVELS)
    H = size(Y, 1)
    reg = regime_index(X, σx)
    off = zeros(Float32, H, 2, length(τ), 2)
    for g in 1:2, c in 1:2, h in 1:H
        r = Y[h, c, reg .== g] .- point[h, c, reg .== g]
        isempty(r) && continue
        off[h, c, :, g] = quantile(r, τ)
    end
    return EmpiricalQuantiles(τ, off)
end

"1 for quiet windows, 2 for active ones (same gate as the physics loss)."
regime_index(X, σx) = ifelse.(vec(regime_gates(X, σx, 0.15f0)[1]) .> 0.5f0, 1, 2)

"Apply static error bars to a point forecast: (H, 2, Q, N)."
function apply_quantiles(e::EmpiricalQuantiles, point::Array{Float32,3}, X::Array{Float32,3}, σx::Float32)
    reg = regime_index(X, σx)
    out = Array{Float32,4}(undef, size(point, 1), 2, length(e.τ), size(point, 3))
    for n in axes(point, 3)
        out[:, :, :, n] = reshape(point[:, :, n], size(point, 1), 2, 1) .+ e.offsets[:, :, :, reg[n]]
    end
    return out
end

"""
    exceed_prob(qv, τ, thr)

P(value ≥ thr) from quantile values `qv` (sorted) at levels `τ`: the CDF is piecewise linear through the quantiles
and extrapolated linearly to 0 and 1 beyond the outer ones.
"""
function exceed_prob(qv::AbstractVector, τ::Vector{Float32}, thr)
    Q = length(τ)
    lo = qv[1] - (qv[2] - qv[1]) * τ[1] / (τ[2] - τ[1])
    hi = qv[Q] + (qv[Q] - qv[Q-1]) * (1 - τ[Q]) / (τ[Q] - τ[Q-1])
    thr <= lo && return 1.0
    thr >= hi && return 0.0
    xs = (lo, qv..., hi)
    ps = (0f0, τ..., 1f0)
    for k in 1:length(xs)-1
        if thr <= xs[k+1]
            a, b = xs[k], xs[k+1]
            F = b > a ? ps[k] + (ps[k+1] - ps[k]) * (thr - a) / (b - a) : ps[k+1]
            return 1.0 - F
        end
    end
    return 0.0
end

"Brier score and Brier skill score vs a constant base-rate forecast `clim`."
function brier(p::AbstractVector, o::AbstractVector{Bool}, clim::Real)
    bs = mean(abs2, p .- o)
    ref = mean(abs2, clim .- o)
    return (; bs, bss = 1 - bs / ref)
end

"Reliability table: mean forecast probability vs observed frequency in `nbins` equal-width bins."
function reliability(p::AbstractVector, o::AbstractVector{Bool}; nbins::Int = 10)
    edges = range(0, 1; length = nbins + 1)
    rows = NamedTuple[]
    for b in 1:nbins
        sel = (p .>= edges[b]) .& (b == nbins ? p .<= 1 : p .< edges[b+1])
        n = count(sel)
        push!(rows, (; bin = b, n, p_mean = n > 0 ? mean(p[sel]) : NaN, o_freq = n > 0 ? mean(o[sel]) : NaN))
    end
    return DataFrame(rows)
end

"""
    score_quantiles(q, w, s; τ, clim_m, clim_s1)

Proper scores in physical units for quantile forecasts `q` (H, 2, Q, N) of windows `w`:
pinball loss (dex), coverage and width of the central 50% / 90% intervals, by channel and regime, and Brier skill
for "≥M1 within the next hour" and "≥10 pfu (S1) within the next hour". Event probability over the horizon is
max over lead times of P(exceed), a lower bound on the true joint probability.
"""
function score_quantiles(q::Array{Float32,4}, w::Windows, s::Standardizer; τ::Vector{Float32} = QUANTILE_LEVELS,
                         clim_m::Real, clim_s1::Real)
    H, _, Q, N = size(q)
    lq = similar(q); ly = similar(w.Y)
    for c in 1:2
        lq[:, c, :, :] = q[:, c, :, :] .* s.σ[c] .+ s.μ[c]
        ly[:, c, :] = w.Y[:, c, :] .* s.σ[c] .+ s.μ[c]
    end
    reg = regime_index(w.X, s.σ[1])
    i05, i25, i75, i95 = (findfirst(≈(t), τ) for t in (0.05f0, 0.25f0, 0.75f0, 0.95f0))
    cover(c, a, b, sel) = mean((ly[:, c, sel] .>= lq[:, c, a, sel]) .& (ly[:, c, sel] .<= lq[:, c, b, sel]))
    width(c, a, b, sel) = mean(lq[:, c, b, sel] .- lq[:, c, a, sel])
    pin(c, sel) = pinball_channel(lq[:, c, :, sel], ly[:, c, sel], τ)
    allw, qu, ac = trues(N), reg .== 1, reg .== 2

    # event probabilities over the horizon
    pm = [maximum(exceed_prob(view(lq, h, 1, :, n), τ, -5.0) for h in 1:H) for n in 1:N]
    ps = [maximum(exceed_prob(view(lq, h, 2, :, n), τ, 1.0) for h in 1:H) for n in 1:N]
    om = vec(maximum(ly[:, 1, :]; dims = 1)) .>= -5
    os = vec(maximum(ly[:, 2, :]; dims = 1)) .>= 1
    new = vec(w.X[end, 1, :]) .* s.σ[1] .+ s.μ[1] .< -5
    bm, bs1 = brier(pm, om, clim_m), brier(ps, os, clim_s1)
    bmn = brier(pm[new], om[new], clim_m)
    return (; pinball_x = pin(1, allw), pinball_p = pin(2, allw),
            pinball_x_quiet = pin(1, qu), pinball_x_active = pin(1, ac),
            cover90_x = cover(1, i05, i95, allw), cover50_x = cover(1, i25, i75, allw),
            cover90_x_quiet = cover(1, i05, i95, qu), cover90_x_active = cover(1, i05, i95, ac),
            cover90_p = cover(2, i05, i95, allw), cover50_p = cover(2, i25, i75, allw),
            width90_x = width(1, i05, i95, allw), width90_x_quiet = width(1, i05, i95, qu),
            width90_x_active = width(1, i05, i95, ac), width90_p = width(2, i05, i95, allw),
            brier_m = bm.bs, bss_m = bm.bss, bss_m_new = bmn.bss, brier_s1 = bs1.bs, bss_s1 = bs1.bss,
            prob_m = pm, obs_m = om, prob_s1 = ps, obs_s1 = os)
end
