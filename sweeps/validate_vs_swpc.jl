# Check the NCEI loader against SWPC's operational product for the same satellite and week:
# XRS-B should match closely; the reconstructed ≥10 MeV protons should track SWPC's ≥10 MeV integral channel.
#     julia -t 1 --project=. sweeps/validate_vs_swpc.jl
include(joinpath(@__DIR__, "..", "sol_setup.jl"))

swpc = load_swpc_cache()                         # primary satellite in the cached week = GOES-18
d0, d1 = Date(first(swpc.time)) + Day(1), Date(last(swpc.time)) - Day(1)
ncei = load_ncei(18, d0, d1)
j = innerjoin(select(swpc, :time, :xray => :x_swpc, :proton => :p_swpc),
              select(ncei, :time, :xray => :x_ncei, :proton => :p_ncei); on = :time)
lx = log10.(j.x_ncei) .- log10.(j.x_swpc)
lp = log10.(j.p_ncei) .- log10.(j.p_swpc)
q(v) = round.(quantile(v, [0.05, 0.5, 0.95]); digits = 3)
println("Matched minutes: ", nrow(j), " (", d0, " → ", d1, ")")
println("XRS-B  log10(NCEI/SWPC) 5/50/95%: ", q(lx), "   corr(log) ", round(cor(log10.(j.x_ncei), log10.(j.x_swpc)); digits = 4))
println("≥10MeV log10(NCEI/SWPC) 5/50/95%: ", q(lp), "   corr(log) ", round(cor(log10.(j.p_ncei), log10.(j.p_swpc)); digits = 4))
println("SWPC ≥10 MeV range ", extrema(j.p_swpc), " ; reconstructed range ", extrema(j.p_ncei))

with_theme(SolarFlareForecast.base_theme()) do
    fig = Figure(size = (1000, 520))
    d = Dates.value.(j.time .- first(j.time)) ./ 86_400_000
    ax1 = Axis(fig[1, 1]; yscale = log10, ylabel = "XRS-B (W m⁻²)", title = "GOES-18: NCEI L2 science data vs SWPC real-time", xticklabelsvisible = false)
    ax2 = Axis(fig[2, 1]; yscale = log10, ylabel = "≥10 MeV (pfu)", xlabel = "Days since $(Date(first(j.time)))")
    linkxaxes!(ax1, ax2)
    lines!(ax1, d, j.x_swpc; color = "#8a8984", linewidth = 1.5, label = "SWPC real-time")
    lines!(ax1, d, j.x_ncei; color = "#2a78d6", linewidth = 0.8, label = "NCEI L2 (quality-filtered)")
    lines!(ax2, d, j.p_swpc; color = "#8a8984", linewidth = 1.5, label = "SWPC ≥10 MeV integral")
    lines!(ax2, d, j.p_ncei; color = "#eb6834", linewidth = 0.8, label = "reconstructed from SGPS spectrum")
    axislegend(ax1; position = :rt); axislegend(ax2; position = :rt)
    save(joinpath(FIG_DIR, "ncei_vs_swpc.png"), fig; px_per_unit = 2)
end
println("Figure → figures/ncei_vs_swpc.png")
