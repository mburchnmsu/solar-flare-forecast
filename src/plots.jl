# CairoMakie diagnostics. One y-scale per panel; fixed colour per model across every figure.

const MODEL_COLORS = Dict(
    "Persistence" => "#8a8984",
    "CNN (MSE only)" => "#2a78d6",
    "CNN + naive smoothing" => "#eb6834",
    "CNN + physics loss" => "#1baf7a",
)
const INK = "#0b0b0b"
const INK_2 = "#52514e"

function base_theme()
    Theme(fontsize = 13, textcolor = INK,
          Axis = (xgridcolor = (:black, 0.06), ygridcolor = (:black, 0.06), spinewidth = 0.8,
                  xticklabelcolor = INK_2, yticklabelcolor = INK_2, titlealign = :left,
                  titlefont = :regular, titlesize = 14),
          Legend = (framevisible = false, labelsize = 12))
end

"Overview of the telemetry with GOES class levels, the S1 threshold, and the train/val/test split."
function plot_telemetry(df::AbstractDataFrame, rows; path = nothing)
    with_theme(base_theme()) do
        fig = Figure(size = (1100, 560))
        days = df.minute ./ 1440
        ax1 = Axis(fig[1, 1]; yscale = log10, ylabel = "XRS-B flux (W m⁻²)", title = "GOES-like 0.1–0.8 nm X-ray flux",
                   xticklabelsvisible = false)
        ax2 = Axis(fig[2, 1]; yscale = log10, ylabel = "≥10 MeV protons (pfu)", xlabel = "Day",
                   title = "Integral proton flux")
        linkxaxes!(ax1, ax2)
        for (ax, split_rows) in ((ax1, rows), (ax2, rows))
            vspan!(ax, days[first(rows.val)], days[last(rows.val)]; color = (:black, 0.04))
            vspan!(ax, days[first(rows.test)], days[last(rows.test)]; color = (:black, 0.08))
        end
        lines!(ax1, days, df.xray; color = "#2a78d6", linewidth = 0.6)
        for (letter, lvl) in pairs(XRAY_CLASS_FLOOR)
            letter in (:C, :M, :X) || continue
            hlines!(ax1, lvl; color = INK_2, linestyle = :dash, linewidth = 0.8)
            text!(ax1, days[end], lvl; text = String(letter), align = (:left, :center), offset = (6, 0), color = INK_2)
        end
        lines!(ax2, days, max.(df.proton, PROTON_FLOOR_PFU); color = "#eb6834", linewidth = 0.8)
        hlines!(ax2, SEP_THRESHOLD_PFU; color = INK_2, linestyle = :dash, linewidth = 0.8)
        text!(ax2, days[end], SEP_THRESHOLD_PFU; text = "S1", align = (:left, :center), offset = (6, 0), color = INK_2)
        ylims!(ax1, 1e-8, 1e-3)
        text!(ax1, days[first(rows.val)], 3e-4; text = "val", offset = (4, 0), color = INK_2)
        text!(ax1, days[first(rows.test)], 3e-4; text = "test", offset = (4, 0), color = INK_2)
        xlims!(ax1, 0, days[end] * 1.03)
        path === nothing || save(path, fig; px_per_unit = 2)
        fig
    end
end

"Validation MSE per epoch for each trained variant."
function plot_training(histories::Vector{<:Pair}; path = nothing)
    with_theme(base_theme()) do
        fig = Figure(size = (700, 380))
        ax = Axis(fig[1, 1]; xlabel = "Epoch", ylabel = "Validation MSE (standardised)", yscale = log10,
                  title = "Validation loss")
        for (name, h) in histories
            scatterlines!(ax, h.epoch, h.val_mse; color = MODEL_COLORS[name], label = name,
                          linewidth = 2, markersize = 8)
        end
        axislegend(ax; position = :rt)
        path === nothing || save(path, fig; px_per_unit = 2)
        fig
    end
end

"RMSE (dex, vs noise-free truth) as a function of forecast lead time, one panel per channel."
function plot_skill_vs_horizon(results::Vector{<:Pair}; path = nothing)
    with_theme(base_theme()) do
        fig = Figure(size = (1000, 380))
        ax1 = Axis(fig[1, 1]; xlabel = "Lead time (min)", ylabel = "RMSE (dex)", title = "X-ray flux")
        ax2 = Axis(fig[1, 2]; xlabel = "Lead time (min)", ylabel = "RMSE (dex)", title = "≥10 MeV protons")
        for (name, r) in results
            c = MODEL_COLORS[name]
            ls = name == "Persistence" ? :dash : :solid
            lines!(ax1, 1:length(r.rmse_x_h), r.rmse_x_h; color = c, linewidth = 2, linestyle = ls, label = name)
            lines!(ax2, 1:length(r.rmse_p_h), r.rmse_p_h; color = c, linewidth = 2, linestyle = ls)
        end
        Legend(fig[1, 3], ax1)
        path === nothing || save(path, fig; px_per_unit = 2)
        fig
    end
end

"""
    plot_forecast_examples(cases, w, s, preds, Zclean; path)

`cases` is a vector of (title, window index); each panel shows the last 2 h of input, the noisy observation,
the noise-free truth, and every model's 60-min X-ray forecast.
"""
function plot_forecast_examples(cases, w::Windows, s::Standardizer, preds::Vector{<:Pair},
                                Yclean::Array{Float32,3}; path = nothing, history_min = 120,
                                truth_label = "Truth (noise-free)")
    with_theme(base_theme()) do
        fig = Figure(size = (1100, 330 * cld(length(cases), 2)))
        W, H = size(w.X, 1), size(w.Y, 1)
        th = (-history_min + 1):0
        tf = 1:H
        local ax
        for (k, (title, n)) in enumerate(cases)
            ax = Axis(fig[cld(k, 2), mod1(k, 2)]; title, xlabel = "Minutes from forecast time",
                      ylabel = "log₁₀ XRS-B (W m⁻²)")
            lines!(ax, th, to_log10(s, w.X[W-history_min+1:W, 1, n], 1); color = INK_2, linewidth = 1, label = "Observed")
            lines!(ax, tf, to_log10(s, w.Y[:, 1, n], 1); color = (INK_2, 0.5), linewidth = 1)
            lines!(ax, tf, to_log10(s, Yclean[:, 1, n], 1); color = INK, linewidth = 2, linestyle = :dot,
                   label = truth_label)
            for (name, ŷ) in preds
                lines!(ax, tf, to_log10(s, ŷ[:, 1, n], 1); color = MODEL_COLORS[name], linewidth = 2, label = name)
            end
            vlines!(ax, 0; color = (:black, 0.25), linewidth = 1)
        end
        Legend(fig[cld(length(cases), 2) + 1, 1:2], ax; orientation = :horizontal, nbanks = 1)
        path === nothing || save(path, fig; px_per_unit = 2)
        fig
    end
end
