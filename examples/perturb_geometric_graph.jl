# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

"""
    perturb_geometric_graph.jl

Example script demonstrating how to use CalibratedTemperedPowerLaw with DynamicGeometricGraphs
to apply controlled noise to geometric graphs and measure spectral distances.

Usage:
    julia --project=. examples/perturb_geometric_graph.jl
"""

using CalibratedTemperedPowerLaw
using DynamicGeometricGraphs
using Graphs
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using StaticArrays

# Helper function to build adjacency matrix from DynamicGeometricGraph
function build_adjacency_matrix(g::DynamicGeometricGraph)
    n = Graphs.nv(g)
    I = Int[]
    J = Int[]
    V = Float64[]

    # Get the vertex IDs
    M, vids = DynamicGeometricGraphs.get_coords(g)
    vid_to_idx = Dict(vid => i for (i, vid) in enumerate(vids))

    # Iterate through edges
    for (u_vid, neighbors) in g.edges
        if haskey(vid_to_idx, u_vid)
            u_idx = vid_to_idx[u_vid]
            for (v_vid, _) in neighbors
                if haskey(vid_to_idx, v_vid)
                    v_idx = vid_to_idx[v_vid]
                    push!(I, u_idx)
                    push!(J, v_idx)
                    push!(V, 1.0)  # Binary adjacency
                end
            end
        end
    end

    return sparse(I, J, V, n, n)
end

println("=" ^ 70)
println("CTPL + DynamicGeometricGraphs Integration Example")
println("=" ^ 70)
println()

# Step 1: Create a reference geometric graph
println("1. Creating reference graph...")
ctr_spoke = 300.0
motif_distance = 100.0
g_ref = refgraph(ctr_spoke, motif_distance)
println("   ✓ Created graph with $(nv(g_ref)) vertices and $(ne(g_ref)) edges")
println()

# Step 2: Auto-tune noise parameters
println("2. Auto-tuning noise parameters...")
c = 1.0
limit = 40.0
p_tail = 0.05  # Target: P(displacement > 40) ≈ 0.05

println("   Searching for λ such that P(X > $limit) ≈ $p_tail...")
λ = autotune_lambda(c, limit, p_tail; tol=0.01, n_samples=2000, max_iter=30)
println("   ✓ Found λ = $λ")
println()

# Step 3: Verify the tuned parameters
println("3. Verifying tuned parameters...")
Random.seed!(42)
verification_samples = [sample_tempered_levy(c, λ; limit=Inf) for _ in 1:1000]
empirical_tail_prob = count(x -> x > limit, verification_samples) / length(verification_samples)
println("   Empirical tail probability: $(round(empirical_tail_prob, digits=4))")
println("   Target tail probability:    $p_tail")
println("   Error: $(round(abs(empirical_tail_prob - p_tail), digits=4))")
println()

# Step 4: Perturb the graph
println("4. Perturbing graph coordinates...")
Random.seed!(42)

# Make a deep copy to preserve original
g_noisy = freeze(g_ref)

# Get coordinate accessor functions
get_coords(g) = DynamicGeometricGraphs.get_coords(g)
function update_coord!(g, vid, coord)
    M, vids = get_coords(g)
    idx = findfirst(==(vid), vids)
    old_coord = StaticArrays.SVector{2, Float64}(M[idx, :])
    new_coord = StaticArrays.SVector{2, Float64}(coord)
    DynamicGeometricGraphs.update_coord!(g, old_coord, new_coord)
end

# Apply perturbation
perturb_graph(g_noisy, get_coords, update_coord!; λ=λ, c=c, limit=limit, seed=42)

println("   ✓ Applied CTPL noise to all vertices")
println()

# Step 5: Compute displacements
println("5. Analyzing coordinate displacements...")
M_ref, vids = get_coords(g_ref)
M_noisy, _ = get_coords(g_noisy)

displacements = [norm(M_noisy[i, :] - M_ref[i, :]) for i in 1:size(M_ref, 1)]

println("   Displacement statistics:")
println("   - Mean:   $(round(mean(displacements), digits=2))")
println("   - Median: $(round(median(displacements), digits=2))")
println("   - Std:    $(round(std(displacements), digits=2))")
println("   - Min:    $(round(minimum(displacements), digits=2))")
println("   - Max:    $(round(maximum(displacements), digits=2))")
println("   - All bounded by limit ($limit): $(all(d <= limit for d in displacements))")
println()

# Step 6: Compute spectral distance
println("6. Computing spectral distance...")

A_ref = build_adjacency_matrix(g_ref)
A_noisy = build_adjacency_matrix(g_noisy)

_, λ_ref = supra_spectrum(A_ref; mode=:comb)
_, λ_noisy = supra_spectrum(A_noisy; mode=:comb)

ref_lambda = maximum(λ_ref)
spectral_dist = w2_1d_empirical(λ_ref, λ_noisy; ref_lambda=ref_lambda)

println("   Reference graph spectrum range: [$(round(minimum(λ_ref), digits=2)), $(round(maximum(λ_ref), digits=2))]")
println("   Noisy graph spectrum range:     [$(round(minimum(λ_noisy), digits=2)), $(round(maximum(λ_noisy), digits=2))]")
println("   Normalized Wasserstein-2 distance: $(round(spectral_dist, digits=6))")
println()

# Step 7: Multiple perturbations to assess variability
println("7. Assessing noise model variability (10 runs)...")
n_runs = 10
spectral_distances = Float64[]

Random.seed!(123)
for run in 1:n_runs
    g_tmp = freeze(g_ref)
    perturb_graph(g_tmp, get_coords, update_coord!; λ=λ, c=c, limit=limit)

    A_tmp = build_adjacency_matrix(g_tmp)
    _, λ_tmp = supra_spectrum(A_tmp; mode=:comb)

    dist = w2_1d_empirical(λ_ref, λ_tmp; ref_lambda=ref_lambda)
    push!(spectral_distances, dist)
end

println("   Spectral distance statistics over $n_runs runs:")
println("   - Mean:   $(round(mean(spectral_distances), digits=6))")
println("   - Median: $(round(median(spectral_distances), digits=6))")
println("   - Std:    $(round(std(spectral_distances), digits=6))")
println("   - Min:    $(round(minimum(spectral_distances), digits=6))")
println("   - Max:    $(round(maximum(spectral_distances), digits=6))")
println()

# Summary
println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println("Successfully demonstrated CTPL graph perturbation:")
println("  • Created geometric graph with $(nv(g_ref)) vertices")
println("  • Auto-tuned λ = $(round(λ, digits=4)) for tail control")
println("  • Applied bounded CTPL noise (limit = $limit)")
println("  • Computed spectral distances using Wasserstein-2")
println("  • Verified consistent noise model behavior")
println()
println("The CTPL noise model provides:")
println("  ✓ Controlled perturbation magnitudes")
println("  ✓ Heavy-tailed but bounded displacements")
println("  ✓ Tunable parameters for different noise regimes")
println("  ✓ Spectral distance quantification")
println("=" ^ 70)
