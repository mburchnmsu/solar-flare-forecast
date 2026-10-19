# Training loop with mini-batch Adam, gradient clipping, and early stopping on validation MSE.

"Forecast a whole split in batches; returns (H, 2, N) in standardised units."
function predict(model, X::Array{Float32,3}; batchsize::Int = 4096)
    N = size(X, 3)
    out = Array{Float32,3}(undef, model.H, 2, N)
    for i in 1:batchsize:N
        j = min(i + batchsize - 1, N)
        out[:, :, i:j] = model(X[:, :, i:j])
    end
    return out
end

"""
    train_model(data, loss; epochs, batchsize, lr, seed, patience) -> (model, history)

Same seed ⇒ same initial weights and batch order for every loss variant, so ablations differ only in the loss.
The checkpoint with the lowest validation MSE (data fit only, identical for all variants) is restored at the end.
"""
function train_model(data, loss::PhysicsLoss; epochs::Int = 30, batchsize::Int = 256, lr = 1f-3,
                     seed::Int = 1, patience::Int = 6, verbose::Bool = true, label = "model",
                     init_state = nothing)
    rng = Xoshiro(seed)
    model = build_model(data.W, data.H; rng)
    init_state === nothing || Flux.loadmodel!(model, init_state)       # fine-tune from a saved model
    opt = Flux.setup(OptimiserChain(ClipNorm(1f0), Adam(Float32(lr))), model)
    loader = Flux.DataLoader((data.train.X, data.train.Y); batchsize, shuffle = true, rng)
    σ = data.scaler.σ
    history = DataFrame(epoch = Int[], train_loss = Float64[], val_mse = Float64[], seconds = Float64[])
    best_mse, best_state, stall = Inf, Flux.state(model), 0
    for epoch in 1:epochs
        t = @elapsed begin
            running, nb = 0.0, 0
            for (xb, yb) in loader
                l, g = Flux.withgradient(m -> total_loss(loss, m(xb), yb, xb, σ), model)
                Flux.update!(opt, model, g[1])
                running += l
                nb += 1
            end
            val_mse = mean(abs2, predict(model, data.val.X) .- data.val.Y)
        end
        push!(history, (epoch, running / nb, val_mse, t))
        verbose && @printf("[%s] epoch %2d  train %.5f  val MSE %.5f  (%.1fs)\n", label, epoch, running / nb, val_mse, t)
        verbose && flush(stdout)
        if val_mse < best_mse
            best_mse, best_state, stall = val_mse, deepcopy(Flux.state(model)), 0
        else
            (stall += 1) >= patience && break
        end
    end
    Flux.loadmodel!(model, best_state)
    return model, history
end
