# 1D-CNN forecaster with a persistence skip connection, and the physics-informed loss.

"""
    ResidualForecaster(net, H)

Predicts the next `H` minutes of both channels as *corrections to persistence*: ŷ = x_last + net(x).
The network only has to learn how the flux will change, and an untrained model already equals the
persistence baseline, which is the standard reference forecast in operational space weather.
"""
struct ResidualForecaster{C}
    net::C
    H::Int
end
Flux.@layer ResidualForecaster trainable = (net,)

function (m::ResidualForecaster)(x::AbstractArray{Float32,3})
    last = x[end:end, 1:2, :]                                  # (1, 2, B)
    return reshape(m.net(x), m.H, 2, size(x, 3)) .+ last       # (H, 2, B)
end

"Conv stack: 3 input channels, receptive field over the whole window, output 2H values."
function build_model(W::Int, H::Int; rng::AbstractRNG = Xoshiro(1), width::Int = 32)
    W % 12 == 0 || throw(ArgumentError("window length W must be divisible by 12 (got $W)"))
    init = Flux.glorot_uniform(rng)
    net = Chain(
        Conv((7,), 3 => width ÷ 2, gelu; pad = SamePad(), init),
        MaxPool((2,)),
        Conv((5,), width ÷ 2 => width, gelu; pad = SamePad(), init),
        MaxPool((2,)),
        Conv((5,), width => width, gelu; pad = SamePad(), init),
        MaxPool((3,)),
        Flux.flatten,
        Dense((W ÷ 12) * width => 128, gelu; init),
        Dense(128 => 2H; init = Flux.zeros32),                 # start exactly at persistence
    )
    return ResidualForecaster(net, H)
end

"""
    PhysicsLoss(; λ_decay, λ_rough, λ_convex, λ_naive, τ_cool_min, τ_sep_min, quiet_dex)

Total loss = MSE + Σ λᵢ·Pᵢ, every penalty evaluated on the forecast in physical log10 units (dex), with the
last observed sample prepended so the jump from "now" into the forecast is constrained too.

* `decay`  — **cooling bound.** A flaring loop cannot fade faster than its fastest radiative/conductive
  cooling time, so d(log10 F)/dt ≥ -1/(τ_cool ln 10). Rises are left free: the impulsive phase can be
  arbitrarily fast. Protons get the same one-sided bound with the much longer `τ_sep_min`, since SEP fluxes
  fade over hours as particles escape and diffuse.
* `rough`  — **quiescent curvature.** Squared second difference of the X-ray forecast, weighted by a soft
  "quiet" gate computed from the input window (30-min range < `quiet_dex`). This kills high-frequency
  artifacts on a quiet background but does not damp genuine flare rises. Protons are always smooth at 1-min.
* `convex` — **gradual-phase shape.** During a decay (input below its recent peak and falling), the log flux of
  an exponentially cooling flare over a background is convex: the decline slows toward the background.
  Concave bends are penalised.
* `naive`  — the uniform first-difference penalty from the original prototype, kept only as an ablation.
"""
Base.@kwdef struct PhysicsLoss
    λ_decay::Float32 = 0f0
    λ_rough::Float32 = 0f0
    λ_convex::Float32 = 0f0
    λ_naive::Float32 = 0f0
    τ_cool_min::Float32 = 5f0
    τ_sep_min::Float32 = 180f0
    quiet_dex::Float32 = 0.15f0
end

"Soft regime gates from the input window alone (no gradient): (quiet, decaying), each (1, B)."
function regime_gates(x::AbstractArray{Float32,3}, σx::Float32, quiet_dex::Float32)
    W = size(x, 1)
    recent = x[W-29:W, 1, :]
    range_dex = (maximum(recent; dims = 1) .- minimum(recent; dims = 1)) .* σx
    quiet = sigmoid.((quiet_dex .- range_dex) ./ 0.03f0)
    now_ = x[W:W, 1, :]
    below_peak = (maximum(x[W-59:W, 1, :]; dims = 1) .- now_) .* σx
    slope = (now_ .- x[W-4:W-4, 1, :]) .* σx
    decaying = Float32.((below_peak .> 0.1f0) .& (slope .< 0f0))
    return quiet, decaying
end

"Physics penalty values (dex² units) for a batch forecast `ŷ` (H, 2, B) given its inputs `x`."
function physics_terms(L::PhysicsLoss, ŷ, x, σ::Vector{Float32})
    quiet, decaying = ChainRulesCore.ignore_derivatives() do
        regime_gates(x, σ[1], L.quiet_dex)
    end
    zx = vcat(x[end:end, 1, :], ŷ[:, 1, :]) .* σ[1]          # (H+1, B), dex
    zp = vcat(x[end:end, 2, :], ŷ[:, 2, :]) .* σ[2]
    dx = zx[2:end, :] .- zx[1:end-1, :]                       # dex / min
    dp = zp[2:end, :] .- zp[1:end-1, :]
    rx = 1f0 / (L.τ_cool_min * log(10f0))
    rp = 1f0 / (L.τ_sep_min * log(10f0))
    decay = mean(abs2, relu.(-dx .- rx)) + mean(abs2, relu.(-dp .- rp))
    # curvature only inside the forecast: the last observation is noisy and the forecast should not chase it
    cx = dx[3:end, :] .- dx[2:end-1, :]                       # dex / min²
    cp = dp[3:end, :] .- dp[2:end-1, :]
    rough = mean(quiet .* abs2.(cx)) + mean(abs2, cp)
    convex = mean(decaying .* abs2.(relu.(-cx)))
    naive = mean(abs2, ŷ[2:end, :, :] .- ŷ[1:end-1, :, :])
    return (; decay, rough, convex, naive)
end

function total_loss(L::PhysicsLoss, ŷ, y, x, σ::Vector{Float32})
    mse = mean(abs2, ŷ .- y)
    p = physics_terms(L, ŷ, x, σ)
    return mse + L.λ_decay * p.decay + L.λ_rough * p.rough + L.λ_convex * p.convex + L.λ_naive * p.naive
end
