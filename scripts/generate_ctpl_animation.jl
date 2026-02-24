# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

"""
    generate_ctpl_animation.jl

Generate an animation showing how the tempering parameter λ affects
the tempered Lévy distribution, demonstrating progressive tail suppression.
Shows historical traces with fade effect.

Usage:
    julia --project=. scripts/generate_ctpl_animation.jl
"""

using CalibratedTemperedPowerLaw
using Plots
using Printf

println("Generating CTPL animation...")

# Set up plot defaults
gr()
default(fontfamily="Computer Modern", linewidth=3, framestyle=:box, 
        grid=false, size=(700, 450), dpi=100)

# Define x range for plotting
x = range(0.01, 50.0, length=500)

# Fixed parameters
c = 1.0
μ = 0.0

# Lambda values to animate - extended range
λ_values = vcat(
    range(0.0, 0.05, length=10),
    range(0.05, 0.15, length=12),
    range(0.15, 0.30, length=10)
)

println("Animating $(length(λ_values)) frames...")

# Pre-compute all PDFs for fade effect
all_pdfs = []
for λ in λ_values
    pdf_vals = [tempered_levy_pdf(xi, c, λ, μ) for xi in x]
    push!(all_pdfs, pdf_vals)
end

anim = @animate for (i, λ) in enumerate(λ_values)
    # Create base plot
    p = plot(xlims=(0, 50),
             ylims=(0, 0.25),
             xlabel="x",
             ylabel="Probability Density",
             title="Tempered Lévy Distribution",
             legend=false)
    
    # Plot all previous curves with fade
    for j in 1:i-1
        plot!(x, all_pdfs[j], 
              color=:steelblue,
              alpha=0.1,
              linewidth=1.5,
              label="")
    end
    
    # Plot current curve highlighted
    plot!(x, all_pdfs[i], 
          color=:steelblue,
          linewidth=3,
          label="")
    
    # Add lambda annotation
    annotate!(35, 0.22, 
              text(@sprintf("λ = %.3f", λ), 16, :right, :steelblue, :bold))
    
    # Add explanation text
    if λ < 0.05
        annotate!(25, 0.18, 
                  text("Heavy tails", 12, :center, :gray))
    elseif λ < 0.15
        annotate!(25, 0.18, 
                  text("Moderate suppression", 12, :center, :gray))
    else
        annotate!(25, 0.18, 
                  text("Strong tail control", 12, :center, :gray))
    end
    
    # Show vertical line at limit
    vline!([40.0], color=:red, linestyle=:dash, linewidth=2, alpha=0.3, label="")
    
    p
end

gif(anim, "docs/ctpl_animation.gif", fps=4)
println("✓ Saved to docs/ctpl_animation.gif")
