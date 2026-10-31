using Test, SolarFlareForecast, Flux, Statistics, DataFrames

@testset "SolarFlareForecast" begin
    df, ev = generate_telemetry(TelemetryConfig(n_days = 5, seed = 7))
    @test nrow(df) == 5 * 1440
    @test all(df.xray .> 0) && all(df.proton .> 0)
    @test flare_class(2.3e-5) == "M2.3"

    data = prepare_data(df; W = 60, H = 12, train_stride = 5)
    @test size(data.train.X)[1:2] == (60, 3) && size(data.train.Y)[1:2] == (12, 2)
    @test first(data.rows.val) - last(data.rows.train) > 60 + 12          # guard gap: no shared minutes

    m = build_model(60, 12)
    x = data.test.X[:, :, 1:8]
    @test m(x) ≈ persistence(Windows(x, data.test.Y[:, :, 1:8], data.test.t0[1:8]))  # untrained = persistence

    L = PhysicsLoss(λ_decay = 1f3, λ_rough = 1f3, λ_convex = 1f3)
    p = physics_terms(L, m(x), x, data.scaler.σ)
    @test p.decay == 0 && p.rough == 0 && p.convex == 0                   # a flat forecast is physical
    drop = m(x) .- reshape(Float32.(0.5 .* (1:12) ./ data.scaler.σ[1]), 12, 1, 1) .* [1f0, 0f0]'
    @test physics_terms(L, drop, x, data.scaler.σ).decay > 0               # 0.5 dex/min drop violates cooling

    g = Flux.gradient(mm -> total_loss(L, mm(x), data.test.Y[:, :, 1:8], x, data.scaler.σ), m)[1]
    @test g.net.layers[end].weight !== nothing                             # gradients reach the network

    trained, _ = train_model(data, PhysicsLoss(); epochs = 1, verbose = false)
    ft, _ = train_model(data, PhysicsLoss(); epochs = 0, verbose = false, init_state = Flux.state(trained),
                        reset_proton_head = true)
    pers = persistence(data.test)
    @test predict(ft, data.test.X)[:, 2, :] == pers[:, 2, :]                 # proton head back at persistence
    @test predict(ft, data.test.X)[:, 1, :] == predict(trained, data.test.X)[:, 1, :]  # X-ray head kept

    @test exceed_prob(Float32[-1, -0.5, 0, 0.5, 1], QUANTILE_LEVELS, 0.0) ≈ 0.5 atol = 1e-6   # median
    @test exceed_prob(Float32[-1, -0.5, 0, 0.5, 1], QUANTILE_LEVELS, 1.0) ≈ 0.05 atol = 1e-6  # 95th percentile
    F, P = trunk_features(trained, data.test.X)
    @test P ≈ predict(trained, data.test.X)                                  # features reproduce the point forecast
    q = predict_quantiles(build_quantile_head(12), F, P)
    @test all(diff(q; dims = 3) .>= 0) && q[:, :, 3, :] ≈ P                # ordered, median starts at the point forecast
end
