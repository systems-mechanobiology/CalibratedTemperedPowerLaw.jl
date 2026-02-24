# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

#!/usr/bin/env julia

# Simple local documentation builder for CalibratedTemperedPowerLaw.jl
# Usage: julia docs/build_local.jl

println("Building CalibratedTemperedPowerLaw.jl documentation locally...")
println("="^60)

cd(@__DIR__)

# Check if we can load the package
println("\n1. Loading CalibratedTemperedPowerLaw package...")
try
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
    using CalibratedTemperedPowerLaw
    println("   CalibratedTemperedPowerLaw loaded successfully")
catch e
    println("   Error loading CalibratedTemperedPowerLaw: $e")
    println("   Make sure you're in the CalibratedTemperedPowerLaw.jl directory")
    exit(1)
end

# Check for Documenter
println("\n2. Checking for Documenter.jl...")
try
    using Documenter
    println("   Documenter.jl available")
catch e
    println("   Documenter.jl not found")
    println("   Installing Documenter.jl...")
    import Pkg
    Pkg.add("Documenter")
    using Documenter
end

# Build documentation
println("\n3. Building documentation...")
try
    makedocs(
        sitename = "CalibratedTemperedPowerLaw.jl",
        format = Documenter.HTML(
            prettyurls = false,  # Local build, no pretty URLs
            assets = String[],
        ),
        modules = [CalibratedTemperedPowerLaw],
        pages = [
            "Home" => "index.md",
            "User Guide" => [
                "Quick Start" => "quickstart.md",
                "SC & S3I Guide" => "sc_s3i.md",
            ],
            "API Reference" => [
                "Core Functions" => "api/core.md",
                "Noise Quantification" => "api/sc_s3i_api.md",
            ],
            "Examples" => "examples.md",
        ],
        checkdocs = :none,
        clean = true,
        doctest = false,
    )
    println("   Documentation built successfully")
catch e
    println("   Error building documentation: $e")
    exit(1)
end

println("\n" * "="^60)
println("Documentation built successfully!")
println("Open: file://$(joinpath(@__DIR__, "build", "index.html"))")
println("="^60)
