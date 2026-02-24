# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

#!/usr/bin/env julia
#
# Visualise how the tempered Levy distribution compares to Gaussian and
# standard Levy, and how the tempering parameter lambda controls the tail.
#
# Output: scripts/noise_comparison.png

using CairoMakie
using CalibratedTemperedPowerLaw: levy_pdf, tempered_levy_pdf

# ---------- evaluation grid ----------
xs = range(0.01, 12.0, length=1000)

# ---------- distributions ----------
# Gaussian (μ=1, σ=1) shifted to positive support for visual comparison
gaussian(x) = (1 / sqrt(2pi)) * exp(-(x - 1.0)^2 / 2)

# Standard Levy (c=1, no tempering)
levy_vals    = [levy_pdf(x, 1.0) for x in xs]

# Tempered Levy at several lambda values
tl_01 = [tempered_levy_pdf(x, 1.0, 0.05) for x in xs]   # light tempering
tl_05 = [tempered_levy_pdf(x, 1.0, 0.3)  for x in xs]   # moderate
tl_20 = [tempered_levy_pdf(x, 1.0, 1.0)  for x in xs]   # strong

gauss_vals = [gaussian(x) for x in xs]

# ---------- plot ----------
fig = Figure(size=(800, 420))
ax = Axis(fig[1, 1],
    xlabel = "x",
    ylabel = "density (unnormalised)",
    title  = "Tempered Levy interpolates between Levy and Gaussian-like tails",
    yscale = log10,
    limits = (nothing, (1e-6, 1.5)),
)

lines!(ax, collect(xs), gauss_vals, color=:grey,       linewidth=2.5, linestyle=:dash,  label="Gaussian")
lines!(ax, collect(xs), levy_vals,  color=:red,        linewidth=2.5, linestyle=:dash,  label="Levy (no tempering)")
lines!(ax, collect(xs), tl_01,      color=:orange,     linewidth=2,                     label="Tempered Levy  lambda=0.05")
lines!(ax, collect(xs), tl_05,      color=:dodgerblue, linewidth=2,                     label="Tempered Levy  lambda=0.3")
lines!(ax, collect(xs), tl_20,      color=:purple,     linewidth=2,                     label="Tempered Levy  lambda=1.0")

axislegend(ax, position=:rt, labelsize=11)

outpath = joinpath(@__DIR__, "noise_comparison.png")
save(outpath, fig, px_per_unit=2)
println("Saved to $outpath")
