# Experiment G1: the R4 recipe (synthetic pre-training → proton head reset → real fine-tune) trained on GOES-16 AND
# GOES-18 (train 2022–2023 of both), early-stopped on the pooled 2024-01 → 2024-06 validation windows.
#     julia -t 1 --project=. sweeps/train_g18.jl <tag> <seed> <init_run>[-pr] [epochs]
# Output: sweeps/results/g1618_<tag>_ft-<init_run>_s<seed>/run.jls (predictions saved for every test set).
include(joinpath(@__DIR__, "g18.jl"))

tag = ARGS[1]
seed = parse(Int, ARGS[2])
init_run = ARGS[3]
reset_proton_head = endswith(init_run, "-pr")
epochs = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 20
name = "g1618_$(tag)_ft-$(init_run)_s$(seed)"
init_run = chopsuffix(init_run, "-pr")
out = joinpath(@__DIR__, "results", name)
isfile(joinpath(out, "run.jls")) && (println("SKIP ", name); exit())

group, loss = parse_variant(tag)
init_state = deserialize(joinpath(@__DIR__, "results", init_run, "run.jls")).state
_, _, synth, _ = sweep_data()
scaler = synth.scaler
_, d16 = real_data(; scaler)
_, d18 = g18_windows(scaler)
data = (; train = catw(d16.train, d18.train), val = catw(d16.val, d18.val), test = d16.test, scaler, W = 180, H = 60)
println(name, ": train ", length(d16.train), " + ", length(d18.train), ", val ", length(d16.val), " + ", length(d18.val),
        "; tests: G16 ", length(d16.test), ", G18 ", length(d18.test), ", G18 new ", length(d18.new))
flush(stdout)
model, hist = train_model(data, loss; epochs, seed, patience = 5, label = name, init_state, reset_proton_head)
mkpath(out)
lossnt = NamedTuple{fieldnames(PhysicsLoss)}(Tuple(getfield(loss, f) for f in fieldnames(PhysicsLoss)))
serialize(joinpath(out, "run.jls"), (; tag = name, seed, group, loss = lossnt, history = hist, state = Flux.state(model),
                                      scaler, init_run, reset_proton_head, satellites = (16, 18),
                                      pred_test = predict(model, d16.test.X), pred_test_g18 = predict(model, d18.test.X),
                                      pred_new_g18 = predict(model, d18.new.X)))
println("DONE ", name)
