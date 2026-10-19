# Loss variants are named by tag, so any λ can be swept without editing code:
#   mse            plain MSE
#   naive_<λ>      MSE + uniform first-difference penalty (the original prototype's smoothing)
#   phys_<λ>       MSE + cooling bound + quiet curvature + decay-phase convexity, all at weight λ
#   physp_<λ>_<s>  as phys_<λ>, with the proton parts of the penalties scaled by s (per-channel weighting)
#   decay_<λ> | rough_<λ> | convex_<λ>   one physics term alone (ablation)
# λ is chosen on the VALIDATION split only; the test split is scored once, after selection.

const FAMILY = Dict("mse" => "CNN (MSE only)", "naive" => "CNN + naive smoothing", "phys" => "CNN + physics loss",
                    "physp" => "Physics, per-channel λ",
                    "decay" => "Cooling bound only", "rough" => "Quiet curvature only", "convex" => "Decay convexity only")

function parse_variant(tag::AbstractString)
    tag == "mse" && return FAMILY["mse"], PhysicsLoss()
    parts = split(tag, '_')
    fam, λ = parts[1], parse(Float32, parts[2])
    loss = fam == "naive"  ? PhysicsLoss(λ_naive = λ) :
           fam == "phys"   ? PhysicsLoss(λ_decay = λ, λ_rough = λ, λ_convex = λ) :
           fam == "physp"  ? PhysicsLoss(λ_decay = λ, λ_rough = λ, λ_convex = λ, proton_scale = parse(Float32, parts[3])) :
           fam == "decay"  ? PhysicsLoss(λ_decay = λ) :
           fam == "rough"  ? PhysicsLoss(λ_rough = λ) :
           fam == "convex" ? PhysicsLoss(λ_convex = λ) : throw(ArgumentError("unknown variant $tag"))
    return FAMILY[fam], loss
end

"Same data for every variant: seed-42 synthetic telemetry, W = 180, H = 60."
function sweep_data(; data_seed = 42)
    df, events = generate_telemetry(TelemetryConfig(seed = data_seed))
    data = prepare_data(df; W = 180, H = 60)
    Zclean = standardize(data.scaler, log_features(df; xray = :xray_clean, proton = :proton_clean))
    return df, events, data, Zclean
end
