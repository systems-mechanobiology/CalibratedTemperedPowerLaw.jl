# CalibratedTemperedPowerLaw.jl Documentation

This directory contains the documentation for CalibratedTemperedPowerLaw.jl built with [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

## Building Documentation Locally

### Quick Build

```bash
cd CalibratedTemperedPowerLaw.jl
julia docs/build_local.jl
```

This will:
1. Install Documenter.jl if needed
2. Build the documentation
3. Show you the path to open the docs in your browser

### Manual Build

If you prefer to build manually:

```bash
cd CalibratedTemperedPowerLaw.jl/docs
julia --project=.
```

Then in Julia:
```julia
julia> using Pkg
julia> Pkg.instantiate()  # Install dependencies
julia> include("make.jl")  # Build docs
```

The built documentation will be in `docs/build/index.html`.

## Documentation Structure

```
docs/
├── src/                    # Source files
│   ├── index.md           # Homepage
│   ├── quickstart.md      # Getting started guide
│   ├── sc_s3i.md          # Complete SC & S3I guide
│   ├── examples.md        # Worked examples
│   └── api/               # API references
│       ├── core.md        # Core functions
│       └── sc_s3i_api.md  # SC & S3I API
├── build/                 # Generated HTML (gitignored)
├── make.jl               # Documenter build script
├── build_local.jl        # Simple local build script
└── Project.toml          # Docs environment
```

## Viewing Documentation

After building, open the documentation:

**macOS:**
```bash
open docs/build/index.html
```

**Linux:**
```bash
xdg-open docs/build/index.html
```

**Windows:**
```bash
start docs/build/index.html
```

Or simply navigate to the path shown after building.

## Writing Documentation

### Adding Examples

Edit `docs/src/examples.md` to add new worked examples.

### Adding API Documentation

1. Add comprehensive docstrings to functions in `src/CalibratedTemperedPowerLaw.jl`
2. Reference them in `docs/src/api/*.md` using:
   ```markdown
   ```@docs
   function_name
   ```\`\`\`
   ```

### Testing Examples

All code examples in the documentation should be runnable. Test them with:

```julia
using CalibratedTemperedPowerLaw, DynamicGeometricGraphs
# Copy and run examples...
```

## Documentation Style Guide

### Code Examples

- **Always include `using` statements**
- **Use meaningful variable names**
- **Show expected output when helpful**
- **Include error handling for common pitfalls**

Good:
```julia
using CalibratedTemperedPowerLaw, Random

Random.seed!(42)
distances = [0.1, 0.15, 0.12]
SC_vals = SC(distances; ks=[1,2])
println("Mean: $(SC_vals[1])")  # Shows output
```

Bad:
```julia
# Missing using statements
d = [0.1, 0.15]  # Unclear variable
x = f(d)  # What is f?
```

### Docstrings

Follow Julia conventions:
- First line: Brief description
- Arguments section with types
- Returns section
- Examples that work standalone
- See Also for related functions

### Mathematical Notation

Use LaTeX in markdown:
```markdown
```math
S3I = \frac{|\mu_a - \mu_b|}{\sqrt{(\sigma_a^2 + \sigma_b^2)/2}}
```\`\`\`
```

## Contributing Documentation

1. Edit source files in `docs/src/`
2. Build locally to check: `julia docs/build_local.jl`
3. Check that examples run correctly
4. Commit changes (don't commit `docs/build/`)

## Troubleshooting

### "Package not found"

Make sure CalibratedTemperedPowerLaw.jl is in your `LOAD_PATH` or use the build script.

### "Documenter.jl not installed"

```julia
using Pkg
Pkg.add("Documenter")
```

### "Function not documented"

Check that:
1. Function has a docstring in `src/CalibratedTemperedPowerLaw.jl`
2. Function is exported
3. Documenter can find the function

### Missing cross-references

Use `@ref` for internal links:
```markdown
See [SC & S3I Guide](@ref) for details.
```
