# Setup cell for every notebook. Dated notebooks (notebooks/SolMMDDYYYY.ipynb) use:
#     include(joinpath(@__DIR__, "..", "sol_setup.jl"))
# Notebooks at the project root (SolOnboarding.ipynb) use:
#     include("sol_setup.jl")
# Activates this project's environment (Project.toml / Manifest.toml) and loads the pipeline module.
using Pkg
Pkg.activate(@__DIR__; io = devnull)
isfile(joinpath(@__DIR__, "Manifest.toml")) || Pkg.instantiate()

using SolarFlareForecast
using CairoMakie, DataFrames, Dates, Statistics, Random, Printf
const PROJECT_DIR = @__DIR__
const FIG_DIR = joinpath(PROJECT_DIR, "figures")
const RESULTS_DIR = joinpath(PROJECT_DIR, "results")
mkpath.((FIG_DIR, RESULTS_DIR))
println("SolarFlareForecast ready · Julia ", VERSION, " · ", Threads.nthreads(), " threads")
