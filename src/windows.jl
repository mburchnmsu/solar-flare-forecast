# Preprocessing: log10 transform, train-only standardisation, contiguous sliding windows,
# and a chronological train/val/test split with guard gaps so no window straddles two splits.

const PROTON_FLOOR_PFU = 0.05    # avoids log10(0) on detector dropouts

"Per-channel affine map between log10 physical flux and the network's standardised units."
struct Standardizer
    μ::Vector{Float32}
    σ::Vector{Float32}
end

"Log10 feature matrix (T × 2): [XRS-B, >=10 MeV protons]."
log_features(df::AbstractDataFrame; xray = :xray, proton = :proton) =
    hcat(Float32.(log10.(df[!, xray])), Float32.(log10.(max.(df[!, proton], PROTON_FLOOR_PFU))))

fit_standardizer(L::AbstractMatrix{Float32}) =
    Standardizer(vec(mean(L; dims = 1)), vec(std(L; dims = 1)))

standardize(s::Standardizer, L::AbstractMatrix{Float32}) = (L .- s.μ') ./ s.σ'
to_log10(s::Standardizer, Z::AbstractArray{Float32}, ch::Int) = Z .* s.σ[ch] .+ s.μ[ch]

"""
    Windows

Network-ready tensors. `X` is (W, 3, N): standardised log X-ray, log protons, and the 1-minute X-ray
increment (a Neupert-like rate feature). `Y` is (H, 2, N): the next H minutes of both standardised channels.
`t0` is the row index of the last input sample of each window.
"""
struct Windows
    X::Array{Float32,3}
    Y::Array{Float32,3}
    t0::Vector{Int}
end
Base.length(w::Windows) = length(w.t0)

"Indices `i` such that rows i-W+1 … i+H are consecutive minutes (no data gaps) and lie inside `rows`."
function valid_anchors(minute::AbstractVector{<:Integer}, rows::UnitRange{Int}, W::Int, H::Int, stride::Int)
    anchors = Int[]
    for i in (first(rows) + W - 1):stride:(last(rows) - H)
        minute[i+H] - minute[i-W+1] == W + H - 1 && push!(anchors, i)
    end
    return anchors
end

function make_windows(Z::AbstractMatrix{Float32}, anchors::Vector{Int}, W::Int, H::Int)
    N = length(anchors)
    X = Array{Float32,3}(undef, W, 3, N)
    Y = Array{Float32,3}(undef, H, 2, N)
    @inbounds for (n, i) in enumerate(anchors)
        for k in 1:W, c in 1:2
            X[k, c, n] = Z[i-W+k, c]
        end
        X[1, 3, n] = 0f0
        for k in 2:W
            X[k, 3, n] = X[k, 1, n] - X[k-1, 1, n]
        end
        for k in 1:H, c in 1:2
            Y[k, c, n] = Z[i+k, c]
        end
    end
    return Windows(X, Y, anchors)
end

"""
    prepare_data(df; W = 180, H = 60, fractions = (0.70, 0.15), train_stride = 3, eval_stride = 1)

Chronological split → fit standardiser on the training rows only → windows per split.
A guard gap of W + H rows between splits removes look-ahead leakage from overlapping windows.
"""
function prepare_data(df::AbstractDataFrame; W::Int = 180, H::Int = 60, fractions = (0.70, 0.15),
                      train_stride::Int = 3, eval_stride::Int = 1)
    T = nrow(df)
    n_tr = floor(Int, fractions[1] * T)
    n_va = floor(Int, fractions[2] * T)
    gap = W + H
    rows = (train = 1:n_tr, val = (n_tr + gap + 1):(n_tr + n_va), test = (n_tr + n_va + gap + 1):T)

    L = log_features(df)
    scaler = fit_standardizer(L[rows.train, :])
    Z = standardize(scaler, L)
    mk(r, s) = make_windows(Z, valid_anchors(df.minute, r, W, H, s), W, H)
    return (train = mk(rows.train, train_stride), val = mk(rows.val, eval_stride),
            test = mk(rows.test, eval_stride), scaler = scaler, rows = rows, W = W, H = H)
end

"""
    window_with(df, scaler; W = 180, H = 60, stride = 1)

Window a whole series with an *existing* standardiser (e.g. apply a synthetic-trained model to real SWPC data).
"""
function window_with(df::AbstractDataFrame, s::Standardizer; W::Int = 180, H::Int = 60, stride::Int = 1)
    Z = standardize(s, log_features(df))
    return make_windows(Z, valid_anchors(df.minute, 1:nrow(df), W, H, stride), W, H)
end
