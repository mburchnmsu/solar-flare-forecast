# Train one loss variant on REAL GOES-16 data (NCEI L2, 2022-01 → 2025-03), optionally fine-tuning a synthetic model.
#     julia -t 1 --project=. sweeps/train_real.jl <tag> [seed] [init_run] [epochs]
#   init_run: a synthetic run directory name in sweeps/results (e.g. phys_1e4_s1) to fine-tune from; "-" for none.
# Split by date: train 2022–2023, validate 2024-01 → 2024-06, test 2024-07 → 2025-03 (guard gaps of W+H minutes).
# Output: sweeps/results/real_<tag>[_ft-<init_run>]_s<seed>/run.jls
include(joinpath(@__DIR__, "..", "sol_setup.jl"))
include(joinpath(@__DIR__, "variants.jl"))
using Serialization, Flux

const REAL_FILE = joinpath(SolarFlareForecast.NCEI_DIR, "merged_g16_2022-01-01_2025-03-31.jls")
const SPLIT_TIMES = (DateTime(2024, 1, 1), DateTime(2024, 7, 1))

"Real-data windows. Train stride 5 / eval stride 5 keeps memory near 1 GB per worker."
function real_data(; scaler = nothing)
    df = deserialize(REAL_FILE)
    data = prepare_data(df; W = 180, H = 60, split_times = SPLIT_TIMES, train_stride = 5, eval_stride = 5)
    scaler === nothing && return df, data
    # fine-tuning: re-standardise with the synthetic model's scaler so its weights see familiar inputs
    Z = standardize(scaler, log_features(df))
    mk(r, s) = SolarFlareForecast.make_windows(Z, SolarFlareForecast.valid_anchors(df.minute, r, 180, 60, s), 180, 60)
    return df, (; train = mk(data.rows.train, 5), val = mk(data.rows.val, 5), test = mk(data.rows.test, 5),
                scaler, rows = data.rows, W = 180, H = 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    tag = ARGS[1]
    seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
    init_run = length(ARGS) > 2 ? ARGS[3] : "-"
    epochs = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 20
    name = "real_$(tag)" * (init_run == "-" ? "" : "_ft-$(init_run)") * "_s$(seed)"
    out = joinpath(@__DIR__, "results", name)
    isfile(joinpath(out, "run.jls")) && (println("SKIP ", name); exit())

    group, loss = parse_variant(tag)
    init_state, scaler = nothing, nothing
    if init_run != "-"
        init_state = deserialize(joinpath(@__DIR__, "results", init_run, "run.jls")).state
        _, _, synth, _ = sweep_data()
        scaler = synth.scaler
    end
    df, data = real_data(; scaler)
    println(name, ": train ", length(data.train), ", val ", length(data.val), ", test ", length(data.test), " windows")
    model, hist = train_model(data, loss; epochs, seed, patience = 5, label = name, init_state)
    mkpath(out)
    lossnt = NamedTuple{fieldnames(PhysicsLoss)}(Tuple(getfield(loss, f) for f in fieldnames(PhysicsLoss)))
    serialize(joinpath(out, "run.jls"), (; tag = name, seed, group, loss = lossnt, history = hist,
                                          state = Flux.state(model), scaler = data.scaler, init_run,
                                          pred_val = predict(model, data.val.X), pred_test = predict(model, data.test.X)))
    println("DONE ", name)
end
