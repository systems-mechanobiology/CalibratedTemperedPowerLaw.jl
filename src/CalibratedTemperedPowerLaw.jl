# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

"""
    CalibratedTemperedPowerLaw

A Julia package for Calibrated Tempered Power Law (CTPL) noise modeling and graph perturbation.

This package implements tempered Lévy distributions for modeling noise in graphs,
including functions for:
- Sampling from tempered Lévy distributions
- Auto-tuning noise parameters
- Perturbing graph coordinates with CTPL noise
- Computing spectral distances between graphs
- Stochastic Co-spectrality (SC) metrics for noise observability
- Stochastic Spectral Separation Index (S3I) for comparing noise regimes
"""
module CalibratedTemperedPowerLaw

using Random
using Statistics
using Distributions
using LinearAlgebra
using LinearAlgebra: Diagonal, Symmetric, eigen, opnorm, I, issymmetric
using Graphs
using StaticArrays
using DynamicGeometricGraphs
using DynamicGeometricGraphs: get_coords, update_coord!, freeze, get_vertex_coords

export levy_pdf, tempered_levy_pdf, sample_tempered_levy
export autotune_lambda
export levy_perturb
export supra_spectrum, w2_1d_empirical
export perturb_graph
export SC, strong_distances
export S3I_from_moments, S3I_from_SC, sc_moment_stats
export cantelli_test, bootstrap_s3i_test
export spectral_shift_inf, spectral_shift_l2, weyl_bound, frobenius_bound
export tightness_weyl, tightness_frobenius

"""
    levy_pdf(x, c, μ=0.0)

Compute the probability density function of a standard Lévy distribution.

The Lévy distribution is a continuous probability distribution with support on [μ, ∞).
It is a special case of the inverse-gamma distribution and is characterized by heavy tails.

# Arguments
- `x::Real`: The point at which to evaluate the PDF
- `c::Real`: The scale parameter (must be positive)
- `μ::Real`: The location parameter (default: 0.0)

# Returns
- `Float64`: The PDF value at x

# Mathematical Definition
For x > μ:
```
f(x; c, μ) = √(c / (2π)) * exp(-c / (2(x - μ))) / (x - μ)^(3/2)
```

# Examples
```julia
julia> levy_pdf(1.0, 1.0, 0.0)
0.24197072451914337

julia> levy_pdf(0.5, 1.0, 0.0)  # x <= μ returns 0
0.0
```

# References
- Nolan, J. P. (2020). "Univariate Stable Distributions"
"""
function levy_pdf(x, c, μ=0.0)
    if x <= μ
        return 0.0
    else
        if (x - μ) == 0
            return Inf
        end
        denom = 2 * π * (x - μ)^3
        return sqrt(c / denom) * exp(-c / (2 * (x - μ)))
    end
end

"""
    tempered_levy_pdf(x, c, λ, μ=0.0)

Compute the probability density function of a tempered Lévy distribution.

Tempering exponentially suppresses the heavy tail of the Lévy distribution,
making it more suitable for modeling bounded perturbations. The tempering
parameter λ controls the rate of exponential decay.

# Arguments
- `x::Real`: The point at which to evaluate the PDF
- `c::Real`: The scale parameter (must be positive)
- `λ::Real`: The tempering parameter (must be non-negative; λ=0 gives standard Lévy)
- `μ::Real`: The location parameter (default: 0.0)

# Returns
- `Float64`: The tempered PDF value at x

# Mathematical Definition
```
f_tempered(x; c, λ, μ) = f_levy(x; c, μ) * exp(-λ(x - μ))
```

# Examples
```julia
julia> tempered_levy_pdf(1.0, 1.0, 0.1, 0.0)
0.21871996628264254

julia> tempered_levy_pdf(1.0, 1.0, 0.0, 0.0) ≈ levy_pdf(1.0, 1.0, 0.0)
true
```

# References
- Rosinski, J. (2007). "Tempering stable processes"
"""
function tempered_levy_pdf(x, c, λ, μ=0.0)
    return levy_pdf(x, c, μ) * exp(-λ * (x - μ))
end

"""
    sample_tempered_levy(c, λ; μ=0.0, limit=Inf)

Sample a random variate from a tempered Lévy distribution using rejection sampling.

Uses the standard Lévy distribution as a proposal distribution and accepts samples
according to the tempering factor exp(-λ(x - μ)). The optional limit parameter
truncates the distribution, capping values at the specified limit.

# Arguments
- `c::Real`: The scale parameter (must be positive)
- `λ::Real`: The tempering parameter (must be non-negative)
- `μ::Real`: The location parameter (default: 0.0)
- `limit::Real`: Maximum value to return (default: Inf)

# Returns
- `Float64`: A random sample from the tempered Lévy distribution

# Algorithm
1. Generate candidate x from Lévy(μ, c) via inverse transform: x = μ + c/z² where z ~ N(0,1)
2. Compute acceptance ratio: α = exp(-λ(x - μ))
3. Accept with probability α, otherwise repeat

# Examples
```julia
julia> Random.seed!(42)
julia> sample_tempered_levy(1.0, 0.1; μ=0.0)
4.123456789  # Example output

julia> # Truncated sampling
julia> sample_tempered_levy(1.0, 0.1; μ=0.0, limit=10.0)
7.89  # Always ≤ 10.0
```

# Performance Note
Rejection sampling efficiency decreases as λ increases. For λ > 1, consider
alternative sampling methods.
"""
function sample_tempered_levy(c, λ; μ=0.0, limit=Inf, rng::AbstractRNG=Random.default_rng())
    @assert λ >= 0 "Tempering parameter λ must be non-negative."

    while true
        # Step 1: Generate a candidate sample from standard Lévy(μ, c)
        # Using inverse transform: X = μ + c/Z² where Z ~ N(0,1)
        z = randn(rng)
        x_candidate = μ + c / z^2

        # Step 2: Acceptance ratio is analytically exp(-λ(x - μ))
        # (tempered_levy_pdf / levy_pdf = exp(-λ(x-μ)))
        acceptance_ratio = exp(-λ * (x_candidate - μ))

        # Step 3: Accept or reject the sample
        if rand(rng) < acceptance_ratio
            return min(x_candidate, limit)
        end
    end
end

"""
    autotune_lambda(c, limit, k=0.05; mu=0.0, tol=1e-4, max_iter=100, n_samples=10000)

Automatically tune the tempering parameter λ for a tempered Lévy distribution.

Uses binary search to find the λ value such that P(X > limit) ≈ k, where X ~ TemperedLevy(c, λ, μ).
This is useful for controlling the tail behavior of the distribution to ensure that
perturbations exceed the specified limit with probability approximately k.

# Arguments
- `c::Real`: The scale parameter of the Lévy distribution
- `limit::Real`: The threshold value for tail probability
- `k::Real`: Target tail probability P(X > limit) (default: 0.05)
- `mu::Real`: Location parameter (default: 0.0)
- `tol::Real`: Tolerance for convergence (default: 1e-4)
- `max_iter::Int`: Maximum number of iterations (default: 100)
- `n_samples::Int`: Number of Monte Carlo samples per iteration (default: 10000)

# Returns
- `Float64`: The tuned λ value

# Algorithm
Binary search over λ ∈ [0, 2]:
1. For current λ, estimate P(X > limit) via Monte Carlo
2. If P(X > limit) > k, increase λ (suppress tail)
3. If P(X > limit) < k, decrease λ (enhance tail)
4. Repeat until |P(X > limit) - k| < tol

# Examples
```julia
julia> # Find λ such that P(X > 40) ≈ 0.05
julia> λ = autotune_lambda(1.0, 40.0, 0.05; mu=0.0)
0.0234375

julia> # More stringent tail control
julia> λ = autotune_lambda(5.0, 20.0, 0.01; mu=0.0)
0.125
```

# Performance
Each iteration requires n_samples evaluations. For faster tuning, reduce n_samples
at the cost of accuracy. Typical convergence occurs in 10-20 iterations.

# Warnings
Emits a warning if convergence is not achieved within max_iter iterations.
"""
function autotune_lambda(c, limit, k=0.05; mu=0.0, tol=1e-4, max_iter=100, n_samples=10000)
    # Adaptive bracket: expand lambda_high until p_tail < k
    lambda_low = 0.0
    lambda_high = 2.0

    for _ in 1:20  # at most 20 doublings → lambda_high up to 2^21 ≈ 2M
        count = 0
        for _ in 1:n_samples
            x = sample_tempered_levy(c, lambda_high; μ=mu, limit=Inf)
            if x > limit
                count += 1
            end
        end
        p_tail = count / n_samples
        if p_tail < k
            break  # bracket is valid: p_tail(lambda_high) < k
        end
        lambda_high *= 2.0
    end

    found = false
    lambda_mid = (lambda_low + lambda_high) / 2

    for iter in 1:max_iter
        lambda_mid = (lambda_low + lambda_high) / 2

        # Estimate tail probability by simulation
        count = 0
        for _ in 1:n_samples
            x = sample_tempered_levy(c, lambda_mid; μ=mu, limit=Inf)
            if x > limit
                count += 1
            end
        end
        p_tail = count / n_samples

        # Sampling noise: standard error of p_tail estimate
        stderr = sqrt(max(p_tail * (1 - p_tail), 0.0) / n_samples)

        # Use an effective tolerance that respects Monte-Carlo noise. This
        # avoids spurious non-convergence warnings when `tol` is much smaller
        # than the sampling error. We choose a conservative multiplier (3σ).
        tol_eff = max(tol, 3.0 * stderr)

        # Check convergence using the effective tolerance
        if abs(p_tail - k) < tol_eff
            found = true
            break
        elseif p_tail > k
            # Too much probability in tail, increase λ to suppress it
            lambda_low = lambda_mid
        else
            # Too little probability in tail, decrease λ to enhance it
            lambda_high = lambda_mid
        end
    end

    if !found
        @warn "autotune_lambda did not converge after $max_iter iterations. Best λ=$lambda_mid, target k=$k"
    end

    return lambda_mid
end

"""
    levy_perturb(vx, λ=1, c=1; limit=Inf, seed=nothing)

Perturb a 2D coordinate vector with CTPL noise.

Applies a random displacement sampled from a tempered Lévy distribution with
uniform angular distribution. The magnitude of the displacement follows
TemperedLevy(c, λ), and the direction is uniformly random on [0, 2π).

# Arguments
- `vx::AbstractVector`: Input 2D coordinate vector
- `λ::Real`: Tempering parameter (default: 1)
- `c::Real`: Scale parameter (default: 1)
- `limit::Real`: Maximum displacement magnitude (default: Inf)
- `seed::Union{Nothing,Int}`: Random seed for reproducibility (default: nothing)

# Returns
- `Vector{Float64}`: Perturbed 2D coordinate

# Mathematical Definition
```
v' = v + ρ * [sin(θ), cos(θ)]
```
where ρ ~ TemperedLevy(c, λ) and θ ~ Uniform(0, 2π)

# Examples
```julia
julia> v = [1.0, 2.0]
julia> levy_perturb(v, 0.5, 1.0; limit=10.0, seed=42)
2-element Vector{Float64}:
 3.456
 4.789

julia> # Reproducible perturbation
julia> levy_perturb([0.0, 0.0], 0.1, 1.0; seed=123)
```

# Usage in Graph Perturbation
Typically used to perturb vertex coordinates in geometric graphs:
```julia
perturbed_coords = [levy_perturb(v, λ, c; limit=L) for v in vertices]
```
"""
function levy_perturb(vx, λ=1, c=1; limit=Inf, seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Random.Xoshiro(seed)

    # Sample displacement magnitude from tempered Lévy
    ρ = sample_tempered_levy(c, λ; limit=limit, rng=rng)

    # Sample uniform angle
    θ = 2π * rand(rng)

    # Apply displacement
    return vx .+ ρ .* [sin(θ), cos(θ)]
end

"""
    supra_spectrum(A::AbstractMatrix{<:Real}; mode::Symbol=:norm, isolated::Symbol=:chung,
                   atol::Real=1e-12, clamp_eps::Real=sqrt(eps(Float64)),
                   return_matrix::Bool=true, return_vectors::Bool=false)

Compute the Laplacian spectrum of a graph from its adjacency matrix.

Constructs either the normalized or combinatorial Laplacian and computes its
eigenvalues (and optionally eigenvectors). Handles isolated vertices according
to specified conventions.

# Arguments
- `A::AbstractMatrix`: Symmetric adjacency matrix (n×n)
- `mode::Symbol`: Laplacian type, either `:norm` (normalized) or `:comb` (combinatorial)
- `isolated::Symbol`: How to handle isolated vertices (`:chung` or `:identity`)
- `atol::Real`: Tolerance for symmetry check (default: 1e-12)
- `clamp_eps::Real`: Threshold for clamping near-zero eigenvalues (default: √eps)
- `return_matrix::Bool`: Whether to return the Laplacian matrix (default: true)
- `return_vectors::Bool`: Whether to return eigenvectors (default: false)

# Returns
Depending on `return_matrix` and `return_vectors`:
- `(L, λ)`: Laplacian matrix and eigenvalues
- `(L, λ, v)`: Laplacian matrix, eigenvalues, and eigenvectors
- `λ`: Only eigenvalues

# Laplacian Definitions

## Normalized Laplacian (:norm)
```
L_norm = I - D^(-1/2) A D^(-1/2)
```
where D is the degree matrix. For isolated vertices (degree 0), the `:chung`
convention sets the corresponding row/column and diagonal to zero.

## Combinatorial Laplacian (:comb)
```
L_comb = D - A
```

# Examples
```julia
julia> A = [0 1 1; 1 0 1; 1 1 0]  # Triangle graph
julia> L, λ = supra_spectrum(A; mode=:comb)
julia> λ
3-element Vector{Float64}:
 0.0
 3.0
 3.0

julia> # Normalized Laplacian with eigenvectors
julia> L, λ, v = supra_spectrum(A; mode=:norm, return_vectors=true)
```

# Notes
- The normalized Laplacian eigenvalues are in [0, 2] for connected graphs
- Eigenvalue 0 has multiplicity equal to the number of connected components
- Small eigenvalues (< clamp_eps) are clamped to exactly 0.0

# References
- Chung, F. R. K. (1997). "Spectral Graph Theory"
"""
function supra_spectrum(A::AbstractMatrix{<:Real};
                        mode::Symbol = :norm,
                        isolated::Symbol = :chung,
                        atol::Real = 1e-12,
                        clamp_eps::Real = sqrt(eps(Float64)),
                        return_matrix::Bool = true,
                        return_vectors::Bool = false)

    n, m = size(A)
    n == m || error("Adjacency must be square, got $(size(A)).")
    issymmetric(A) || error("Adjacency not symmetric.")

    deg = vec(sum(A, dims=2))
    any(deg .< 0) && error("Negative degrees detected.")

    # Construct Laplacian based on mode
    L = if mode == :norm
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        DinvS = Diagonal(ifelse.(deg .> 0, 1 ./ sqrt.(deg), 0.0))
        I - DinvS * A * DinvS
    elseif mode == :comb
        # Combinatorial Laplacian: L = D - A
        Diagonal(deg) - A
    else
        error("Unknown mode=$(mode). Use :norm or :comb.")
    end

    # Handle isolated vertices for normalized Laplacian
    if mode == :norm
        if isolated == :chung
            z = findall(deg .== 0)
            if !isempty(z)
                L[z, :] .= 0.0
                L[:, z] .= 0.0
                @inbounds for i in z
                    L[i, i] = 0.0
                end
            end
        elseif isolated == :identity
            # Leave L as constructed
        else
            error("isolated must be :chung or :identity")
        end
    end

    # Enforce numerical symmetry
    symdiff = L - L'
    if opnorm(symdiff, Inf) < atol
        L = (L + L') / 2
    else
        error("L is not symmetric (||L-L'||_∞=$(opnorm(symdiff, Inf))).")
    end

    # Compute eigendecomposition
    LS = Symmetric(Matrix(L))
    ev = eigen(LS)
    vals = ev.values
    vecs = ev.vectors

    # Clamp near-zero eigenvalues
    vals = ifelse.(abs.(vals) .< clamp_eps, 0.0, vals)

    # Return based on flags
    if return_matrix
        if return_vectors
            return LS, vals, vecs
        else
            return LS, vals
        end
    else
        return vals
    end
end

"""
    w2_1d_empirical(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                    ref_lambda::Union{Nothing,Real}=nothing,
                    squared::Bool=false,
                    assume_sorted::Bool=false)

Compute the 1D Wasserstein-2 (W₂) distance between two empirical distributions.

Uses the closed-form solution for 1D optimal transport: sort both distributions
and integrate the squared differences. Optionally normalizes by a reference scale.

# Arguments
- `x::AbstractVector`: First empirical distribution (sample points)
- `y::AbstractVector`: Second empirical distribution (sample points)
- `ref_lambda::Union{Nothing,Real}`: Reference scale for normalization (default: nothing)
- `squared::Bool`: If true, return W₂² instead of W₂ (default: false)
- `assume_sorted::Bool`: Skip sorting if inputs are pre-sorted (default: false)

# Returns
- `Float64`: The W₂ distance (or W₂² if squared=true)

# Mathematical Definition
For empirical distributions with n and m points:
```
W₂²(x, y) = ∫₀¹ (F_x⁻¹(u) - F_y⁻¹(u))² du
```
where F⁻¹ are the quantile functions. This reduces to comparing sorted values.

# Normalization
If `ref_lambda` is provided, the distance is normalized:
- If `squared=false`: W₂ / ref_lambda
- If `squared=true`: W₂² / ref_lambda²

# Examples
```julia
julia> x = [1.0, 2.0, 3.0]
julia> y = [1.5, 2.5, 3.5]
julia> w2_1d_empirical(x, y)
0.5

julia> # With normalization
julia> w2_1d_empirical(x, y; ref_lambda=10.0)
0.05

julia> # Pre-sorted input for efficiency
julia> x_sorted = sort(randn(1000))
julia> y_sorted = sort(randn(1000))
julia> w2_1d_empirical(x_sorted, y_sorted; assume_sorted=true)
```

# Algorithm Complexity
- Time: O(n log n + m log m) for sorting (or O(n + m) if assume_sorted=true)
- Space: O(n + m) for sorted copies

# Applications
- Comparing graph Laplacian spectra
- Spectral distance metrics (e.g., d_SD in graph comparison)
- Distribution divergence measures

# References
- Panaretos, V. M. & Zemel, Y. (2019). "Statistical Aspects of Wasserstein Distances"
"""
function w2_1d_empirical(x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real};
                         ref_lambda::Union{Nothing,Real}=nothing,
                         squared::Bool=false,
                         assume_sorted::Bool=false)

    isempty(x) && throw(ArgumentError("x cannot be empty."))
    isempty(y) && throw(ArgumentError("y cannot be empty."))

    # Determine common floating type
    T = ref_lambda === nothing ?
        float(promote_type(eltype(x), eltype(y))) :
        float(promote_type(eltype(x), eltype(y), typeof(ref_lambda)))

    # Sort into floating copies unless already sorted
    xx = assume_sorted ? T.(x) : sort!(collect(T.(x)))
    yy = assume_sorted ? T.(y) : sort!(collect(T.(y)))

    n = length(xx)
    m = length(yy)
    mx = inv(T(n))  # Mass per point in x
    my = inv(T(m))  # Mass per point in y

    # Two-pointer algorithm to compute W₂²
    i = 1
    j = 1
    remx = mx  # Remaining mass at current x point
    remy = my  # Remaining mass at current y point
    cost = zero(T)
    tol = sqrt(eps(T))

    @inbounds while i <= n && j <= m
        # Transfer mass = min(remaining mass in x, remaining mass in y)
        mass = remx < remy ? remx : remy

        # Accumulate squared distance cost
        dx = xx[i] - yy[j]
        cost += mass * (dx * dx)

        # Update remaining masses
        remx -= mass
        remy -= mass

        # Move to next point if current mass exhausted
        if remx <= tol
            i += 1
            remx = (i <= n) ? mx : zero(T)
        end
        if remy <= tol
            j += 1
            remy = (j <= m) ? my : zero(T)
        end
    end

    w2sq = cost
    w2 = squared ? w2sq : sqrt(w2sq)

    # Normalize by reference scale if provided
    if ref_lambda !== nothing
        λ = T(ref_lambda)
        λ > zero(T) || throw(ArgumentError("ref_lambda must be positive."))
        w2 = squared ? (w2 / (λ * λ)) : (w2 / λ)
    end

    return w2
end

"""
    spectral_shift_inf(L::AbstractMatrix, Lp::AbstractMatrix) -> Float64

Compute the maximum per-eigenvalue shift between two symmetric matrices.

Returns S_inf = max_i |λ_i(Lp) - λ_i(L)|, where eigenvalues are sorted in nondecreasing order.

This is the "actual shift" metric used in perturbation analysis. It measures the worst-case
change in any single eigenvalue.

# Mathematical Properties
- S_inf ≤ ||Lp - L||_2 (Weyl's inequality)
- S_inf ≥ 0

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)

# Returns
- `Float64`: Maximum absolute eigenvalue shift

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
S_inf = spectral_shift_inf(L, Lp)
```

# References
- agents/motifs.md: Definition in section 0
"""
function spectral_shift_inf(L::AbstractMatrix, Lp::AbstractMatrix)
    size(L) == size(Lp) || throw(DimensionMismatch("L and Lp must have the same size"))

    # Compute eigenvalues in sorted order
    λ_L = eigvals(Symmetric(Matrix(L)))
    λ_Lp = eigvals(Symmetric(Matrix(Lp)))

    # Maximum per-eigenvalue shift
    return maximum(abs.(λ_Lp .- λ_L))
end

"""
    spectral_shift_l2(L::AbstractMatrix, Lp::AbstractMatrix) -> Float64

Compute the L2 norm of the eigenvalue difference vector.

Returns S_2 = ||λ(Lp) - λ(L)||_2, where eigenvalues are sorted in nondecreasing order.

This is a secondary shift metric that measures aggregate eigenvalue displacement.

# Mathematical Properties
- S_2 ≤ ||Lp - L||_F (Hoffman-Wielandt inequality)
- S_2 ≥ 0
- S_2 ≤ sqrt(n) * S_inf

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)

# Returns
- `Float64`: L2 norm of eigenvalue shifts

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
S_2 = spectral_shift_l2(L, Lp)
```

# References
- agents/motifs.md: Definition in section 0
"""
function spectral_shift_l2(L::AbstractMatrix, Lp::AbstractMatrix)
    size(L) == size(Lp) || throw(DimensionMismatch("L and Lp must have the same size"))

    # Compute eigenvalues in sorted order
    λ_L = eigvals(Symmetric(Matrix(L)))
    λ_Lp = eigvals(Symmetric(Matrix(Lp)))

    # L2 norm of eigenvalue difference
    return norm(λ_Lp .- λ_L)
end

"""
    weyl_bound(L::AbstractMatrix, Lp::AbstractMatrix) -> Float64

Compute the Weyl bound (spectral norm of perturbation matrix).

Returns B_weyl = ||Δ||_2 where Δ = Lp - L, computed as the operator 2-norm.

For symmetric matrices, ||Δ||_2 = max(|λ_min(Δ)|, |λ_max(Δ)|).

# Mathematical Properties
- B_weyl provides an upper bound on S_inf: S_inf ≤ B_weyl (Weyl's inequality)
- B_weyl ≤ B_frob (spectral norm ≤ Frobenius norm)
- B_weyl ≥ 0

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)

# Returns
- `Float64`: Spectral norm of Lp - L

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
B_weyl = weyl_bound(L, Lp)
```

# References
- agents/motifs.md: Definition in section 0
- Weyl's inequality: max_k |λ_k(A+E) - λ_k(A)| ≤ ||E||_2
"""
function weyl_bound(L::AbstractMatrix, Lp::AbstractMatrix)
    size(L) == size(Lp) || throw(DimensionMismatch("L and Lp must have the same size"))

    Δ = Lp - L
    # For symmetric matrices, spectral norm = max absolute eigenvalue
    λ_Δ = eigvals(Symmetric(Matrix(Δ)))
    return maximum(abs.(λ_Δ))
end

"""
    frobenius_bound(L::AbstractMatrix, Lp::AbstractMatrix) -> Float64

Compute the Frobenius norm of the perturbation matrix.

Returns B_frob = ||Δ||_F where Δ = Lp - L, computed as sqrt(sum(Δ.^2)).

# Mathematical Properties
- B_frob provides an upper bound on B_weyl: B_weyl ≤ B_frob
- B_frob provides an upper bound on S_2: S_2 ≤ B_frob (Hoffman-Wielandt)
- B_frob ≥ 0

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)

# Returns
- `Float64`: Frobenius norm of Lp - L

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
B_frob = frobenius_bound(L, Lp)
```

# References
- agents/motifs.md: Definition in section 0
"""
function frobenius_bound(L::AbstractMatrix, Lp::AbstractMatrix)
    size(L) == size(Lp) || throw(DimensionMismatch("L and Lp must have the same size"))

    Δ = Lp - L
    return norm(Δ)  # Frobenius norm is the default for matrices
end

"""
    tightness_weyl(L::AbstractMatrix, Lp::AbstractMatrix; atol::Real=1e-12) -> Float64

Compute the Weyl tightness ratio: how close the actual shift comes to the Weyl bound.

Returns ρ_weyl = S_inf / B_weyl.

A motif "hits Weyl" if ρ_weyl ≈ 1 (within tolerance), meaning the perturbation achieves
the theoretical worst-case eigenvalue shift.

# Special Cases
- If B_weyl = 0 (no perturbation), returns 1.0 by definition

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)
- `atol::Real`: Absolute tolerance for zero-check (default: 1e-12)

# Returns
- `Float64`: Tightness ratio (typically in [0, 1])

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
ρ = tightness_weyl(L, Lp)
if ρ ≥ 1 - 1e-6
    println("Motif hits Weyl bound!")
end
```

# References
- agents/motifs.md: Definition in section 0
"""
function tightness_weyl(L::AbstractMatrix, Lp::AbstractMatrix; atol::Real=1e-12)
    S_inf = spectral_shift_inf(L, Lp)
    B_weyl = weyl_bound(L, Lp)

    # Handle zero perturbation case
    if B_weyl < atol
        return 1.0
    end

    return S_inf / B_weyl
end

"""
    tightness_frobenius(L::AbstractMatrix, Lp::AbstractMatrix; atol::Real=1e-12) -> Float64

Compute the Frobenius tightness ratio: S_2 / B_frob.

Returns ρ_frob = S_2 / B_frob, measuring how efficiently the Frobenius norm is used
by eigenvalue shifts.

# Special Cases
- If B_frob = 0 (no perturbation), returns 1.0 by definition

# Arguments
- `L::AbstractMatrix`: Reference symmetric matrix
- `Lp::AbstractMatrix`: Perturbed symmetric matrix (same size as L)
- `atol::Real`: Absolute tolerance for zero-check (default: 1e-12)

# Returns
- `Float64`: Tightness ratio (typically in [0, 1])

# Examples
```julia
L = [2.0 -1.0; -1.0 2.0]
Lp = [2.1 -1.0; -1.0 2.1]
ρ = tightness_frobenius(L, Lp)
```

# References
- agents/motifs.md: Definition in section 0
"""
function tightness_frobenius(L::AbstractMatrix, Lp::AbstractMatrix; atol::Real=1e-12)
    S_2 = spectral_shift_l2(L, Lp)
    B_frob = frobenius_bound(L, Lp)

    # Handle zero perturbation case
    if B_frob < atol
        return 1.0
    end

    return S_2 / B_frob
end

"""
    perturb_graph(g, get_coords_fn, update_coords_fn!; λ=0.1, c=1, limit=1, seed=nothing)

Perturb all vertex coordinates of a graph using CTPL noise.

Applies `levy_perturb` to each vertex coordinate independently. This is the main
function for generating noisy versions of geometric graphs under the CTPL model.

# Arguments
- `g`: Graph object
- `get_coords_fn::Function`: Function that takes graph g and returns (coords_matrix, vertex_ids)
- `update_coords_fn!::Function`: Function that takes (g, vertex_id, new_coord) and updates in-place
- `λ::Real`: Tempering parameter for CTPL distribution (default: 0.1)
- `c::Real`: Scale parameter for CTPL distribution (default: 1)
- `limit::Real`: Maximum displacement per coordinate (default: 1)
- `seed::Union{Nothing,Int}`: Random seed for reproducibility (default: nothing)

# Returns
- Modified graph with perturbed coordinates

# Example with DynamicGeometricGraphs
```julia
using CTPL
using DynamicGeometricGraphs

# Create reference graph
g = refgraph(100.0, 30)

# Define coordinate accessors
get_coords(g) = DynamicGeometricGraphs.get_coords(g)
function update_coords!(g, vid, new_coord)
    DynamicGeometricGraphs.update_coord!(g, vid, new_coord)
end

# Perturb graph
λ = autotune_lambda(1.0, 40.0, 0.05)
g_noisy = perturb_graph(g, get_coords, update_coords!; λ=λ, c=1, limit=40)
```

# Noise Model
Each vertex v with coordinates (x, y) is perturbed independently:
```
(x', y') = (x, y) + levy_perturb([x, y], λ, c; limit=limit)
```

The CTPL parameters control:
- `c`: Base scale of perturbations
- `λ`: Tail suppression (higher λ = smaller typical displacements)
- `limit`: Hard cap on displacement magnitude

# Notes
- For reproducibility, set `seed` parameter
- Use `autotune_lambda` to calibrate λ based on desired tail probability
- The function modifies a copy of the graph, not the original
"""
function perturb_graph(g, get_coords_fn, update_coords_fn!; λ=0.1, c=1, limit=1, seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Random.Xoshiro(seed)

    # Get vertex coordinates
    M, vs = get_coords_fn(g)

    # Perturb each vertex (pass rng through to avoid global state)
    for (i, vid) in enumerate(vs)
        old_coord = M[i, :]
        ρ = sample_tempered_levy(c, λ; limit=limit, rng=rng)
        θ = 2π * rand(rng)
        new_coord = old_coord .+ ρ .* [sin(θ), cos(θ)]
        update_coords_fn!(g, vid, new_coord)
    end

    return g
end

"""
    perturb_graph(g::DynamicGeometricGraph; λ=0.1, c=1, limit=1, seed=nothing)

Convenience wrapper for perturbing DynamicGeometricGraphs directly.

Returns a tuple (g_perturbed, coords_matrix, vertex_ids) for compatibility with
code that expects this signature.
"""
function perturb_graph(g::DynamicGeometricGraphs.DynamicGeometricGraph; λ=0.1, c=1, limit=1, seed=nothing)
    # Make a copy to avoid modifying the original
    g_copy = freeze(g)

    # Define coordinate accessors for this graph type
    function get_coords_wrapper(g)
        return get_coords(g)
    end

    function update_coords_wrapper!(g, vid, new_coord)
        # Get old coordinates
        old_coord = get_vertex_coords(g, vid)
        # Convert new_coord to SVector if it isn't already
        if !(new_coord isa SVector)
            new_coord = SVector{length(new_coord)}(new_coord)
        end
        # Call DynamicGeometricGraphs.update_coord! with old and new coordinates
        update_coord!(g, old_coord, new_coord)
    end

    # Call the main perturb_graph function
    g_perturbed = perturb_graph(g_copy, get_coords_wrapper, update_coords_wrapper!;
                                λ=λ, c=c, limit=limit, seed=seed)

    # Get final coordinates for return value
    M, vs = get_coords(g_perturbed)

    return (g_perturbed, M, vs)
end

# ============================================================================
# Stochastic Co-spectrality (SC) and Spectral Separation Index (S3I)
# ============================================================================

"""
    SC(G::DynamicGeometricGraph;
       mode::Symbol       = :strong,   # :strong or :weak
       ks                 = [1],
       n_samples::Integer = 100,
       seed::Union{Nothing,Int} = nothing,
       λ::Real,
       c::Real,
       limit::Real)

Compute **Stochastic Co-spectrality (SC)** - the k-th moments of spectral distance under noise.

SC quantifies "how much does noise change the graph spectrum?" by measuring
E[D^k] where D is the spectral distance between the original and perturbed graph.

# Modes

- **:strong** (oracle): SC_k(G; N) = E[d_SD(G, N(G))]
  - Measures distance from known ground truth G
  - More accurate, requires access to noise-free graph
  - Use for: Synthetic graphs, simulations

- **:weak** (observable): SC_k^obs(G; N) = E[d_SD(N_i(G), N_j(G))]
  - Measures distance between noisy observations
  - More realistic, no ground truth needed
  - Use for: Real data, empirical graphs
  - Note: Observable SC ≤ 2×Strong SC (triangle inequality)

# Arguments
- `G`: Reference graph (DynamicGeometricGraph)
- `mode::Symbol`: `:strong` or `:weak` (default: `:strong`)
- `ks::Vector`: Moment orders to compute (default: `[1]`)
- `n_samples::Int`: Monte Carlo samples (default: 100, use 500+ for publication)
- `λ, c, limit`: CTPL noise parameters (see `perturb_graph`)
- `seed::Union{Nothing,Int}`: Random seed for reproducibility (default: nothing)

# Returns
- `Vector{Float64}`: SC moments [SC_k1, SC_k2, ...] in order of `ks`

# Key SC Values
- `SC[1]` = μ = mean spectral distance
- `SC[2] - SC[1]²` = σ² = variance
- Higher moments characterize distribution shape (use `sc_moment_stats`)

# Examples

## Basic Usage
```julia
using CTPL, DynamicGeometricGraphs, Random

Random.seed!(42)
g = refgraph(100.0, 30)

# Compute first 2 moments
SC_vals = SC(g; mode=:strong, ks=[1,2], n_samples=100,
             λ=0.1, c=1.0, limit=10.0)

println("Mean spectral distance: \$(SC_vals[1])")
println("Variance: \$(SC_vals[2] - SC_vals[1]^2)")
```

## Full Statistical Analysis
```julia
# Compute all 4 moments for complete characterization
SC_all = SC(g; mode=:strong, ks=[1,2,3,4], n_samples=200,
            λ=0.1, c=1.0, limit=10.0)

# Extract statistics
stats = sc_moment_stats(SC_all[1], SC_all[2], SC_all[3], SC_all[4])
println("Mean: \$(stats.μ)")
println("Std: \$(sqrt(stats.σ2))")
println("Excess kurtosis: \$(stats.κ_excess)")
```

## Observable Mode (No Ground Truth)
```julia
# When you only have noisy observations
SC_obs = SC(g; mode=:weak, ks=[1], n_samples=100,
            λ=0.1, c=1.0, limit=10.0)

println("Observable noise level: \$(SC_obs[1])")
```

# See Also
- `sc_moment_stats`: Extract mean, variance, kurtosis from SC moments
- `strong_distances`: Generate raw distance samples
- `S3I_from_SC`: Compare two noise regimes
- Full documentation: `docs/src/sc_s3i.md`
"""
function SC(G::DynamicGeometricGraphs.DynamicGeometricGraph;
            mode::Symbol       = :strong,
            ks                 = [1],
            n_samples::Integer = 100,
            seed::Union{Nothing,Int} = nothing,
            λ::Real,
            c::Real,
            limit::Real)

    rng = isnothing(seed) ? Random.default_rng() : Random.Xoshiro(seed)

    if seed === nothing
        @debug "SC: seed not set. Results will not be reproducible. Set seed=<integer> for reproducibility."
    end

    A  = adjacency_matrix(G)
    _, λg, _ = supra_spectrum(A; mode = :comb, return_vectors = true)
    refλ = maximum(λg)

    if mode === :strong
        # D_i = d_SD(G, N_i(G))
        D = Vector{Float64}(undef, n_samples)

        for i in 1:n_samples
            # Derive per-sample seed from rng
            sample_seed = Int(rand(rng, UInt64) % (2^62))
            Gp, _, _ = perturb_graph(G; λ = λ, c = c, limit = limit, seed = sample_seed)
            _, λp, _ = supra_spectrum(adjacency_matrix(Gp); mode = :comb, return_vectors = true)
            D[i] = w2_1d_empirical(λg, λp; ref_lambda = refλ)
        end

        return SC(D; ks = ks)

    elseif mode === :weak
        # noisy spectra λ₁,…,λ_n
        specs = Vector{Vector{Float64}}(undef, n_samples)
        for i in 1:n_samples
            sample_seed = Int(rand(rng, UInt64) % (2^62))
            Gp, _, _ = perturb_graph(G; λ = λ, c = c, limit = limit, seed = sample_seed)
            _, λp, _ = supra_spectrum(adjacency_matrix(Gp); mode = :comb, return_vectors = true)
            specs[i] = λp
        end

        # Accumulate moments directly from upper triangle (no n×n matrix needed)
        n_pairs = n_samples * (n_samples - 1) ÷ 2
        acc = zeros(Float64, length(ks))
        cnt = 0

        @inbounds for i in 1:n_samples-1, j in i+1:n_samples
            d = w2_1d_empirical(specs[i], specs[j]; ref_lambda = refλ)
            if !isnan(d)
                cnt += 1
                for (t, k) in enumerate(ks)
                    acc[t] += d^k
                end
            end
        end

        return cnt == 0 ? fill(NaN, length(ks)) : acc ./ cnt

    else
        error("Unknown mode = $mode; use :strong or :weak")
    end
end

"""
    SC(D::AbstractVector{<:Real}; ks = [1])

Compute SC moments from a vector of pre-computed spectral distances.

Use this when you already have distance measurements and want to compute
their moments without re-running Monte Carlo simulations.

# Arguments
- `D::Vector`: Spectral distances [d_1, d_2, ..., d_n]
- `ks::Vector`: Moment orders to compute (default: `[1]`)

# Returns
- `Vector{Float64}`: SC moments [E[D^k1], E[D^k2], ...] in order of `ks`

# Example
```julia
# You measured these distances from experiments
distances = [0.032, 0.035, 0.031, 0.038, 0.029, 0.033, 0.036, 0.034]

# Compute moments
SC_vals = SC(distances; ks=[1, 2, 3, 4])

# Extract statistics
stats = sc_moment_stats(SC_vals[1], SC_vals[2], SC_vals[3], SC_vals[4])
println("Mean: \$(stats.μ)")
println("Std:  \$(sqrt(stats.σ2))")
```

# See Also
- `SC(G; ...)`: Compute SC via Monte Carlo on graphs
- `sc_moment_stats`: Extract statistics from moments
"""
function SC(D::AbstractVector{<:Real}; ks = [1])
    Df = float.(D)
    out = Vector{Float64}(undef, length(ks))
    for (j, k) in enumerate(ks)
        out[j] = mean(Df .^ k)
    end
    return out
end

"""
    SC(Δ::AbstractMatrix{<:Real}; ks = [1])

Observable (weak) SC_k from a matrix of pairwise distances
Δ_ij = d_SD(N_i(G), N_j(G)), using i < j.

Returns a Vector{Float64} with SC_ks[j] = SC_{ks[j]}^obs.
"""
function SC(Δ::AbstractMatrix{<:Real}; ks = [1])
    n, m = size(Δ)
    @assert n == m "Δ must be square"

    acc = zeros(Float64, length(ks))
    cnt = 0

    @inbounds for i in 1:n-1, j in i+1:n
        d = float(Δ[i, j])
        if !isnan(d)
            cnt += 1
            for (t, k) in enumerate(ks)
                acc[t] += d^k
            end
        end
    end

    return cnt == 0 ? fill(NaN, length(ks)) : acc ./ cnt
end

"""
    S3I_from_moments(μa, σa2, μb, σb2)

Compute **Stochastic Spectral Separation Index (S3I)** - an effect size metric
for comparing two noise regimes.

S3I is analogous to Cohen's d, measuring how distinguishable two distributions are
by their spectral distances. It answers: "Can I tell these noise levels apart?"

# Formula
```
S3I = |μ_a - μ_b| / √((σ_a² + σ_b²)/2)
```

Where μ, σ² are the mean and variance of spectral distances under each regime.

# Arguments
- `μa, μb`: Mean spectral distances for regimes a and b
- `σa2, σb2`: Variance of spectral distances for regimes a and b

# Returns
- `Float64`: S3I value (effect size)

# Interpretation
- **S3I < 0.5**: Noise regimes are very similar (indistinguishable)
- **S3I ∈ [0.5, 0.8)**: Small effect size
- **S3I ∈ [0.8, 2.0)**: Medium effect size
- **S3I ≥ 2.0**: Large effect size (strongly distinguishable)

# Example
```julia
# Statistics from two experimental conditions
μ_control = 0.032
σ²_control = 0.0001

μ_treatment = 0.058
σ²_treatment = 0.0002

# Compute S3I
s3i = S3I_from_moments(μ_control, σ²_control, μ_treatment, σ²_treatment)
println("S3I = \$(round(s3i, digits=3))")

if s3i > 2.0
    println("✓ Conditions are significantly different")
elseif s3i > 0.8
    println("~ Moderate difference")
else
    println("✗ Conditions are indistinguishable")
end
```

# See Also
- `S3I_from_SC`: Compute S3I from SC moments
- `SC`: Compute stochastic co-spectrality
- Full documentation: `docs/src/sc_s3i.md`
"""
function S3I_from_moments(μa::Real, σa2::Real, μb::Real, σb2::Real)
    # Allow for tiny negative variances due to floating-point roundoff.
    # If variance is slightly negative (within tol), clamp to zero. If it's
    # significantly negative, emit an error with diagnostics.
    tol = 1e-12
    if σa2 < 0
        if σa2 > -tol
            @warn "clamping tiny negative variance to zero" σa2=σa2 μa=μa
            σa2 = 0.0
        else
            throw(ArgumentError("σa2 is negative: $(σa2). μa=$(μa)."))
        end
    end
    if σb2 < 0
        if σb2 > -tol
            @warn "clamping tiny negative variance to zero" σb2=σb2 μb=μb
            σb2 = 0.0
        else
            throw(ArgumentError("σb2 is negative: $(σb2). μb=$(μb)."))
        end
    end

    denom = sqrt((σa2 + σb2) / 2)
    return denom == 0 ? 0.0 : abs(μa - μb) / denom
end

"""
    S3I_from_SC(SC1a, SC2a, SC1b, SC2b)

Compute S3I directly from SC moments (convenience wrapper for `S3I_from_moments`).

This is the most common way to compute S3I after running SC computations on two
noise regimes.

# Arguments
- `SC1a, SC2a`: First two SC moments for regime a
- `SC1b, SC2b`: First two SC moments for regime b

# Returns
- `Float64`: S3I value (effect size)

# Conversion
```
μ_a = SC1a
σ²_a = SC2a - SC1a²
μ_b = SC1b
σ²_b = SC2b - SC1b²
S3I = S3I_from_moments(μ_a, σ²_a, μ_b, σ²_b)
```

# Example
```julia
using CTPL, DynamicGeometricGraphs, Random

Random.seed!(42)
g = refgraph(100.0, 30)

# Low noise regime
SC_low = SC(g; mode=:strong, ks=[1,2], n_samples=100,
            λ=0.05, c=1.0, limit=10.0)

# High noise regime
SC_high = SC(g; mode=:strong, ks=[1,2], n_samples=100,
             λ=0.20, c=1.0, limit=10.0)

# Compare regimes
s3i = S3I_from_SC(SC_low[1], SC_low[2], SC_high[1], SC_high[2])

println("S3I = \$(round(s3i, digits=3))")
if s3i > 2.0
    println("✓ Strongly distinguishable noise levels")
end
```

# See Also
- `S3I_from_moments`: Compute from means and variances directly
- `SC`: Compute stochastic co-spectrality
- Full documentation: `docs/src/sc_s3i.md`
"""
function S3I_from_SC(SC1a::Real, SC2a::Real,
                     SC1b::Real, SC2b::Real)
    μa  = SC1a
    σa2 = SC2a - SC1a^2
    μb  = SC1b
    σb2 = SC2b - SC1b^2
    return S3I_from_moments(μa, σa2, μb, σb2)
end

"""
    sc_moment_stats(SC1, SC2, SC3, SC4)

Extract mean, variance, and excess kurtosis from the first four SC moments.

This provides a statistical characterization of the spectral distance distribution
under noise.

# Arguments
- `SC1, SC2, SC3, SC4`: The first four SC moments E[D^1], E[D^2], E[D^3], E[D^4]

# Returns
- `NamedTuple`: `(μ, σ2, κ_excess)`
  - `μ`: Mean = SC1
  - `σ2`: Variance = SC2 - SC1²
  - `κ_excess`: Excess kurtosis (>0 = heavy-tailed, <0 = light-tailed, ≈0 = Gaussian-like)

# Formula
```
μ = SC_1
σ² = SC_2 - SC_1²
κ_excess = (SC_4 - 4·SC_3·SC_1 + 6·SC_2·SC_1² - 3·SC_1⁴) / σ⁴ - 3
```

# Interpretation
- **μ**: Average spectral perturbation magnitude
- **σ²**: Variability of noise impact
- **κ_excess**: Distribution shape
  - > 0: Heavy-tailed (occasional large perturbations)
  - ≈ 0: Normal-like distribution
  - < 0: Light-tailed (perturbations are consistent)

# Example
```julia
using CTPL, DynamicGeometricGraphs

g = refgraph(100.0, 30)

# Compute all 4 moments
SC_vals = SC(g; mode=:strong, ks=[1,2,3,4], n_samples=200,
             λ=0.1, c=1.0, limit=10.0)

# Extract statistics
stats = sc_moment_stats(SC_vals[1], SC_vals[2], SC_vals[3], SC_vals[4])

println("Mean spectral distance: \$(round(stats.μ, digits=4))")
println("Standard deviation: \$(round(sqrt(stats.σ2), digits=4))")
println("Excess kurtosis: \$(round(stats.κ_excess, digits=2))")

if stats.κ_excess > 0
    println("→ Distribution is heavy-tailed")
end
```

# See Also
- `SC`: Compute SC moments
- Full documentation: `docs/src/sc_s3i.md`
"""
function sc_moment_stats(SC1::Real, SC2::Real, SC3::Real, SC4::Real)
    μ  = SC1
    σ2 = SC2 - SC1^2

    if σ2 <= 0
        return (μ = μ, σ2 = σ2, κ_excess = NaN)
    end

    num = SC4 - 4*SC3*SC1 + 6*SC2*SC1^2 - 3*SC1^4
    κ_excess = num / (σ2^2) - 3

    return (μ = μ, σ2 = σ2, κ_excess = κ_excess)
end

"""
    cantelli_test(D_star::Real, SC1::Real, SC2::Real; α::Real=0.05)

Test if an observed spectral distance D★ is noise-observable using Cantelli's inequality.

This implements the observability test from the paper: an observed distance D★ is
considered *noise-observable at level α* if it significantly exceeds what noise alone
would produce.

# Formula

Uses Cantelli's one-sided inequality to test:
```
(D★ - μ_D) / σ_D > √((1-α)/α)
```

where:
- μ_D = SC₁ (mean spectral distance)
- σ_D = √(SC₂ - SC₁²) (standard deviation)
- α ∈ (0,1) is the significance level

# Arguments
- `D_star::Real`: Observed spectral distance between two graphs
- `SC1::Real`: First SC moment (mean distance under noise)
- `SC2::Real`: Second SC moment
- `α::Real`: Significance level (default: 0.05)

# Returns
- `NamedTuple`: `(is_observable, z_score, threshold, p_value_upper_bound)`
  - `is_observable::Bool`: true if D★ is noise-observable at level α
  - `z_score::Float64`: Standardized distance (D★ - μ_D) / σ_D
  - `threshold::Float64`: Critical value √((1-α)/α)
  - `p_value_upper_bound::Float64`: Upper bound on P(D ≥ D★) from Cantelli

# Mathematical Background

Cantelli's inequality states that for a random variable X with mean μ and variance σ²:
```
P(X ≥ μ + t·σ) ≤ 1 / (1 + t²)
```

Setting the RHS equal to α and solving for t gives t = √((1-α)/α).

# Example
```julia
# You observed a distance of 0.15 between two graphs
D_star = 0.15

# From SC measurements under noise:
SC_vals = SC(G; mode=:strong, ks=[1,2], n_samples=200, λ=0.1, c=1.0, limit=10.0)

# Test if this distance is distinguishable from noise
result = cantelli_test(D_star, SC_vals[1], SC_vals[2]; α=0.05)

if result.is_observable
    println("✓ Distance is noise-observable (z = \$(round(result.z_score, digits=2)))")
    println("  P(D ≥ D★) ≤ \$(round(result.p_value_upper_bound, digits=4))")
else
    println("✗ Distance is consistent with noise alone")
end
```

# See Also
- `SC`: Compute stochastic co-spectrality moments
- `bootstrap_s3i_test`: Bootstrap test for S3I significance
"""
function cantelli_test(D_star::Real, SC1::Real, SC2::Real; α::Real=0.05)
    @assert 0 < α < 1 "α must be in (0, 1)"

    μ_D = SC1
    σ²_D = SC2 - SC1^2

    if σ²_D <= 0
        @warn "Variance is non-positive (σ² = $σ²_D). Cannot perform test."
        return (is_observable = false, z_score = NaN, threshold = NaN, p_value_upper_bound = NaN)
    end

    σ_D = sqrt(σ²_D)

    # Standardized score
    z_score = (D_star - μ_D) / σ_D

    # Cantelli threshold: D★ is observable if z > √((1-α)/α)
    threshold = sqrt((1 - α) / α)

    # Upper bound on P(D ≥ D★) from Cantelli's inequality
    # For z_score > 0: P(D ≥ D★) ≤ 1/(1 + z_score²)
    # For z_score ≤ 0: P(D ≥ D★) = 1 (no information)
    p_value_upper_bound = z_score > 0 ? 1.0 / (1.0 + z_score^2) : 1.0

    is_observable = z_score > threshold

    return (
        is_observable = is_observable,
        z_score = z_score,
        threshold = threshold,
        p_value_upper_bound = p_value_upper_bound
    )
end

"""
    bootstrap_s3i_test(s3i_samples::AbstractVector{<:Real}; α::Real=0.05, method::Symbol=:percentile)

Bootstrap test for S3I significance: test if two noise regimes are spectrally separated.

This implements the distribution-agnostic bootstrap test from the paper: if the
(1-α) bootstrap confidence interval excludes zero, the two regimes are deemed
spectrally separated at level α.

# Arguments
- `s3i_samples::Vector`: Bootstrap or repeated S3I estimates
- `α::Real`: Significance level (default: 0.05)
- `method::Symbol`: Confidence interval method (default: `:percentile`)
  - `:percentile`: Standard percentile bootstrap CI
  - `:basic`: Basic bootstrap CI (not yet implemented)
  - `:bca`: Bias-corrected and accelerated (not yet implemented)

# Returns
- `NamedTuple`: `(is_separated, ci_lower, ci_upper, mean_s3i, median_s3i)`
  - `is_separated::Bool`: true if CI excludes zero (regimes are separated)
  - `ci_lower::Float64`: Lower bound of (1-α) confidence interval
  - `ci_upper::Float64`: Upper bound of (1-α) confidence interval
  - `mean_s3i::Float64`: Mean of S3I samples
  - `median_s3i::Float64`: Median of S3I samples

# Algorithm

For the percentile method:
1. Sort the S3I samples
2. Compute quantiles: [α/2, 1-α/2]
3. CI = [S3I_{α/2}, S3I_{1-α/2}]
4. Reject H₀ (no separation) if 0 ∉ CI

# Example

```julia
using CTPL, DynamicGeometricGraphs, Random

Random.seed!(42)
g = refgraph(100.0, 30)

# Generate bootstrap S3I samples by repeated runs
n_bootstrap = 100
s3i_samples = Vector{Float64}(undef, n_bootstrap)

for b in 1:n_bootstrap
    # Low noise regime
    Da = strong_distances(g; n_samples=50, λ=0.05, c=1.0, limit=10.0, seed=b)
    SCa = SC(Da; ks=[1,2])

    # High noise regime
    Db = strong_distances(g; n_samples=50, λ=0.20, c=1.0, limit=10.0, seed=b+1000)
    SCb = SC(Db; ks=[1,2])

    s3i_samples[b] = S3I_from_SC(SCa[1], SCa[2], SCb[1], SCb[2])
end

# Test for separation
result = bootstrap_s3i_test(s3i_samples; α=0.05)

if result.is_separated
    println("✓ Noise regimes are spectrally separated at α=0.05")
    println("  S3I: \$(round(result.mean_s3i, digits=3))")
    println("  95% CI: [\$(round(result.ci_lower, digits=3)), \$(round(result.ci_upper, digits=3))]")
else
    println("✗ Cannot distinguish noise regimes")
end
```

# See Also
- `S3I_from_SC`: Compute S3I from SC moments
- `cantelli_test`: Test if observed distance is noise-observable
"""
function bootstrap_s3i_test(s3i_samples::AbstractVector{<:Real};
                            α::Real=0.05,
                            method::Symbol=:percentile)
    @assert 0 < α < 1 "α must be in (0, 1)"
    @assert !isempty(s3i_samples) "s3i_samples cannot be empty"

    if method != :percentile
        error("Only :percentile method is currently implemented")
    end

    # Remove NaN/Inf values
    valid_samples = filter(isfinite, s3i_samples)

    if isempty(valid_samples)
        @warn "No valid S3I samples. Cannot perform test."
        return (is_separated = false, ci_lower = NaN, ci_upper = NaN,
                mean_s3i = NaN, median_s3i = NaN)
    end

    # Sort for quantile computation
    sorted_samples = sort(valid_samples)

    # Percentile bootstrap CI
    n = length(sorted_samples)
    lower_idx = max(1, floor(Int, n * (α/2)))
    upper_idx = min(n, ceil(Int, n * (1 - α/2)))

    ci_lower = sorted_samples[lower_idx]
    ci_upper = sorted_samples[upper_idx]

    # Test: CI excludes zero?
    is_separated = (ci_lower > 0.0) || (ci_upper < 0.0)

    # Summary statistics
    mean_s3i = mean(valid_samples)
    median_s3i = median(sorted_samples)

    return (
        is_separated = is_separated,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        mean_s3i = mean_s3i,
        median_s3i = median_s3i
    )
end

"""
    strong_distances(G::DynamicGeometricGraph;
                     n_samples::Integer,
                     λ::Real,
                     c::Real,
                     limit::Real,
                     seed::Union{Nothing,Int} = nothing)

Draw `n_samples` i.i.d. strong/oracle spectral distances

    D[i] = d_SD(G, N(G))

under the CTPL vertex-noise model parameterised by (λ, c, limit).

- `G`        : reference graph (DynamicGeometricGraph)
- `λ, c`     : CTPL parameters (tempering, scale)
- `limit`    : truncation / jump limit used in `perturb_graph`
- `seed`     : Random seed for reproducibility (default: nothing)

Returns a Vector{Float64} of length `n_samples`.

Note: Requires DynamicGeometricGraph for spatial perturbation.
"""
function strong_distances(G::DynamicGeometricGraphs.DynamicGeometricGraph;
                          n_samples::Integer,
                          λ::Real,
                          c::Real,
                          limit::Real,
                          seed::Union{Nothing,Int} = nothing)

    rng = isnothing(seed) ? Random.default_rng() : Random.Xoshiro(seed)

    if seed === nothing
        @debug "strong_distances: seed not set. Results will not be reproducible. Set seed=<integer> for reproducibility."
    end

    # reference spectrum once
    A  = adjacency_matrix(G)
    _, λg, _ = supra_spectrum(A; mode = :comb, return_vectors = true)
    refλ = maximum(λg)

    D = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        # Derive per-sample seed from rng to ensure independence + reproducibility
        sample_seed = rand(rng, UInt64)
        Gp, _, _ = perturb_graph(G; λ = λ, c = c, limit = limit, seed = Int(sample_seed % (2^62)))

        # spectrum of noisy graph
        _, λp, _ = supra_spectrum(adjacency_matrix(Gp); mode = :comb, return_vectors = true)

        # one sample of D = d_SD(G, N(G))
        D[i] = w2_1d_empirical(λg, λp; ref_lambda = refλ)
    end

    return D
end

end # module CalibratedTemperedPowerLaw
