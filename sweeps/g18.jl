# GOES-18 data for experiment G1: windows with the same date splits as GOES-16 plus a new out-of-time test period.
#   train 2022-09-13 → 2023-12, val 2024-01 → 2024-06, test 2024-07 → 2025-03 (same period as GOES-16's test),
#   new 2025-04 → 2026-10 (after all GOES-16 data; never used for training or selection).
include(joinpath(@__DIR__, "train_real.jl"))           # real_data(), SPLIT_TIMES, sol_setup
using Serialization, Flux

const G18_FILE = joinpath(SolarFlareForecast.NCEI_DIR, "merged_g18_2022-09-13_2026-10-31.jls")
const NEW_START = DateTime(2025, 4, 1)

"GOES-18 windows standardised with `scaler`: (; train, val, test, new) at a 5-min stride, with guard gaps."
function g18_windows(scaler; W = 180, H = 60, stride = 5)
    df = deserialize(G18_FILE)
    gap = W + H
    n_tr = searchsortedfirst(df.time, SPLIT_TIMES[1]) - 1
    n_va = searchsortedfirst(df.time, SPLIT_TIMES[2]) - 1
    i_new = searchsortedfirst(df.time, NEW_START)
    rows = (train = 1:n_tr, val = (n_tr + gap + 1):n_va, test = (n_va + gap + 1):(i_new - 1),
            new = (i_new + gap):nrow(df))
    Z = standardize(scaler, log_features(df))
    mk(r) = SolarFlareForecast.make_windows(Z, SolarFlareForecast.valid_anchors(df.minute, r, W, H, stride), W, H)
    return df, (; train = mk(rows.train), val = mk(rows.val), test = mk(rows.test), new = mk(rows.new), rows)
end

"Concatenate two window sets (t0 values are row indices of different series; only used for bookkeeping)."
catw(a::Windows, b::Windows) = Windows(cat(a.X, b.X; dims = 3), cat(a.Y, b.Y; dims = 3), vcat(a.t0, b.t0))
