# Evaluation: trajectory errors, physical-consistency diagnostics, and event (threshold) skill.

"Persistence forecast (the operational baseline): the last observed value held for H minutes."
persistence(w::Windows) = repeat(w.X[end:end, 1:2, :], size(w.Y, 1), 1, 1)

"Targets for windows `w` taken from another feature matrix (e.g. the noise-free truth), standardised."
function targets_from(Z::AbstractMatrix{Float32}, w::Windows)
    H = size(w.Y, 1)
    Y = Array{Float32,3}(undef, H, 2, length(w))
    @inbounds for (n, i) in enumerate(w.t0), k in 1:H, c in 1:2
        Y[k, c, n] = Z[i+k, c]
    end
    return Y
end

"True Skill Statistic (Hanssen–Kuipers): POD − POFD. Insensitive to event rarity, standard for flare forecasts."
function tss(pred::AbstractVector{Bool}, obs::AbstractVector{Bool})
    tp = count(pred .& obs);  fn = count(.!pred .& obs)
    fp = count(pred .& .!obs); tn = count(.!pred .& .!obs)
    pod = tp + fn == 0 ? NaN : tp / (tp + fn)
    pofd = fp + tn == 0 ? NaN : fp / (fp + tn)
    return (; tss = pod - pofd, pod, pofd, tp, fn, fp, tn)
end

"""
    evaluate(ŷ, w, scaler; Yclean, loss) -> NamedTuple

All errors in dex (log10 flux). `ŷ` and targets are standardised (H, 2, N).
"""
function evaluate(ŷ::Array{Float32,3}, w::Windows, s::Standardizer; Yclean::Array{Float32,3},
                  loss::PhysicsLoss = PhysicsLoss())
    lx(A) = to_log10(s, A[:, 1, :], 1)
    lp(A) = to_log10(s, A[:, 2, :], 2)
    px, pp = lx(ŷ), lp(ŷ)
    ox, op = lx(w.Y), lp(w.Y)                 # observed (noisy)
    cx, cp = lx(Yclean), lp(Yclean)           # noise-free truth
    quiet, _ = regime_gates(w.X, s.σ[1], loss.quiet_dex)
    q = vec(quiet) .> 0.5f0

    rmse(a, b) = sqrt(mean(abs2, a .- b))
    rmse_h(a, b) = vec(sqrt.(mean(abs2, a .- b; dims = 2)))

    # physical consistency of the X-ray forecast, including the step out of the last observation
    zx = vcat(to_log10(s, w.X[end:end, 1, :], 1), px)
    d = diff(zx; dims = 1)
    curv = diff(diff(px; dims = 1); dims = 1)               # within the forecast only
    rx = 1 / (loss.τ_cool_min * log(10))

    # event skill over the 60-min horizon: M-class X-ray and S1 (≥10 pfu) proton thresholds
    mx_pred, mx_true = vec(maximum(px; dims = 1)), vec(maximum(cx; dims = 1))
    sp_pred, sp_true = vec(maximum(pp; dims = 1)), vec(maximum(cp; dims = 1))
    now_x = vec(to_log10(s, w.X[end:end, 1, :], 1))
    new = now_x .< -5                          # not already at M level: genuinely forward-looking cases
    mclass = tss(mx_pred .>= -5, mx_true .>= -5)
    mclass_new = tss((mx_pred .>= -5)[new], (mx_true .>= -5)[new])
    s1 = tss(sp_pred .>= 1, sp_true .>= 1)

    return (; rmse_x = rmse(px, ox), rmse_p = rmse(pp, op),
            rmse_x_clean = rmse(px, cx), rmse_p_clean = rmse(pp, cp),
            rmse_x_quiet = rmse(px[:, q], cx[:, q]), rmse_x_active = rmse(px[:, .!q], cx[:, .!q]),
            rmse_x_h = rmse_h(px, cx), rmse_p_h = rmse_h(pp, cp),
            rough_quiet = mean(abs, curv[:, q]),
            rough_quiet_truth = mean(abs, diff(diff(cx; dims = 1); dims = 1)[:, q]),
            decay_violation_truth = mean(diff(cx; dims = 1) .< -rx),
            decay_violation_frac = mean(d .< -rx),
            tss_m = mclass.tss, tss_m_new = mclass_new.tss, tss_s1 = s1.tss,
            n_quiet = count(q), n_active = count(!, q), n_m_new_events = mclass_new.tp + mclass_new.fn)
end
