# Train one loss variant with one init seed; save state, history and val/test forecasts.
#     julia -t 1 --project=. sweeps/train_variant.jl <tag> [seed] [epochs]
# Output: sweeps/results/<tag>_s<seed>/run.jls (skipped if it already exists, so chains can resume).
include(joinpath(@__DIR__, "..", "sol_setup.jl"))
include(joinpath(@__DIR__, "variants.jl"))
using Serialization, Flux

tag = ARGS[1]
seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
epochs = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 30
out = joinpath(@__DIR__, "results", "$(tag)_s$(seed)")
if isfile(joinpath(out, "run.jls"))
    println("SKIP ", tag, " seed ", seed, " (already done)")
    exit()
end
group, loss = parse_variant(tag)
df, events, data, Zclean = sweep_data()
model, hist = train_model(data, loss; epochs, seed, patience = 6, label = "$(tag)/s$(seed)")
mkpath(out)
# loss stored as a NamedTuple so later changes to the PhysicsLoss struct never break old run files
lossnt = NamedTuple{fieldnames(PhysicsLoss)}(Tuple(getfield(loss, f) for f in fieldnames(PhysicsLoss)))
serialize(joinpath(out, "run.jls"), (; tag, seed, group, loss = lossnt, history = hist, state = Flux.state(model),
                                      pred_val = predict(model, data.val.X), pred_test = predict(model, data.test.X)))
println("DONE ", tag, " seed ", seed)
