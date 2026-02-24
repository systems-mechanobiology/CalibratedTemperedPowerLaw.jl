# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

"""
    generate_ctpl_figure.jl

Generate a figure demonstrating the CTPL calibration feature:
how different tail probability targets (α) produce different calibrated λ values
for the same maximum jump limit, interpolating between:
- Lévy distribution (heavy tails, α → 1)
- Power law with controlled tails (moderate α)
- Tightly controlled tails (α → 0)

Usage:
    julia --project=. scripts/generate_ctpl_figure.jl
"""

using CalibratedTemperedPowerLaw
using Plots
using Random

# Set up plot defaults
gr()
default(fontfamily="Computer Modern", linewidth=2, framestyle=:box, 
        label=nothing, grid=false, size=(800, 500))

# Define x range for plotting
x = range(0.01, 50.0, length=500)

# Fixed parameters
c = 1.0
μ = 0.0
limit = 40.0  # Same maximum jump for all calibrations

# Calibrate λ for different tail probability targets
println("Calibrating λ for different tail probabilities...")
println("  Fixed: c = $c, max jump = $limit")
println()

# Different tail probability targets
α_high = 0.20    # Allow 20% chance of exceeding limit (heavier tails)
α_medium = 0.05  # Allow 5% chance of exceeding limit (moderate)
α_low = 0.01     # Allow 1% chance of exceeding limit (tight control)

Random.seed!(42)
λ_high = autotune_lambda(c, limit, α_high; tol=0.01, n_samples=2000)
λ_medium = autotune_lambda(c, limit, α_medium; tol=0.01, n_samples=2000)
λ_low = autotune_lambda(c, limit, α_low; tol=0.01, n_samples=2000)

println("  α = $α_high → λ = $(round(λ_high, digits=4)) (heavier tails)")
println("  α = $α_medium → λ = $(round(λ_medium, digits=4)) (moderate)")
println("  α = $α_low → λ = $(round(λ_low, digits=4)) (tight control)")
println()

# Compute PDFs
pdf_levy = [levy_pdf(xi, c, μ) for xi in x]
pdf_high = [tempered_levy_pdf(xi, c, λ_high, μ) for xi in x]
pdf_medium = [tempered_levy_pdf(xi, c, λ_medium, μ) for xi in x]
pdf_low = [tempered_levy_pdf(xi, c, λ_low, μ) for xi in x]

# Create the plot with log scale
p = plot(x, pdf_levy, label="Lévy (λ=0, uncalibrated)", color=:lightgray, 
         linestyle=:dash, linewidth=1.5,
         xlabel="x", ylabel="Probability Density",
         title="Calibrated CTPL: Controlling Tail Probability",
         yscale=:log10, ylims=(1e-5, 10),
         legend=:topright, legendfontsize=9)

plot!(p, x, pdf_high, label="α=0.20, λ=$(round(λ_high, digits=3))", 
      color=:red, linewidth=2.5)
plot!(p, x, pdf_medium, label="α=0.05, λ=$(round(λ_medium, digits=3))", 
      color=:blue, linewidth=2.5)
plot!(p, x, pdf_low, label="α=0.01, λ=$(round(λ_low, digits=3))", 
      color=:green, linewidth=2.5)

# Add vertical line at calibration limit
vline!(p, [limit], color=:black, linestyle=:dot, linewidth=2, 
       label="Max jump = $limit")

# Add annotation
annotate!(p, 8.0, 0.002, text("Calibration:\nP(X > $limit) = α", :black, 9, :left))
annotate!(p, 25.0, 1e-4, text("Larger λ →\nstronger\ntempering", :darkgreen, 8, :center))

# Save the figure
output_path = "docs/ctpl_noise_model.png"
savefig(p, output_path)
println("✓ Figure saved to: $output_path")

# Also create a linear scale version focusing on central region
p2 = plot(x, pdf_levy, label="Lévy (λ=0)", color=:lightgray, 
          linestyle=:dash, linewidth=1.5,
          xlabel="x", ylabel="Probability Density",
          title="Calibrated CTPL: Central Region",
          xlims=(0, 15), ylims=(0, 0.5),
          legend=:topright, legendfontsize=9)

plot!(p2, x, pdf_high, label="α=0.20, λ=$(round(λ_high, digits=3))", 
      color=:red, linewidth=2.5)
plot!(p2, x, pdf_medium, label="α=0.05, λ=$(round(λ_medium, digits=3))", 
      color=:blue, linewidth=2.5)
plot!(p2, x, pdf_low, label="α=0.01, λ=$(round(λ_low, digits=3))", 
      color=:green, linewidth=2.5)

vline!(p2, [limit], color=:black, linestyle=:dot, linewidth=2, 
       label="Max jump = $limit")

# Save the linear scale version
output_path2 = "docs/ctpl_noise_model_linear.png"
savefig(p2, output_path2)
println("✓ Figure saved to: $output_path2")

println("\nFigures demonstrate:")
println("  • Calibrated λ values control tail probability for fixed max jump")
println("  • α = $α_high: Heavy tails, more probability beyond limit")
println("  • α = $α_medium: Moderate tail control (typical use case)")
println("  • α = $α_low: Tight tail control, rare extreme events")
println("  • CTPL calibration enables precise noise regime specification")
