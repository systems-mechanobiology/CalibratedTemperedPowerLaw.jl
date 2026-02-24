# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ben Cardoen <b.cardoen@bham.ac.uk>

using Documenter
using CalibratedTemperedPowerLaw

makedocs(
    sitename = "CalibratedTemperedPowerLaw.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://github.com/bencardoen/CalibratedTemperedPowerLaw.jl",
        assets = String[],
    ),
    modules = [CalibratedTemperedPowerLaw],
    pages = [
        "API Reference" => "index.md",
    ],
    checkdocs = :none,
)

# Only deploy if we're in a CI environment
if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/bencardoen/CalibratedTemperedPowerLaw.jl.git",
        devbranch = "main",
    )
end
