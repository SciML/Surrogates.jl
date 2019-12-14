using Documenter, Surrogates
makedocs(
    sitename="Surrogates.jl",
    pages = [
    "index.md"
    "User guide" => [
        "Samples" => "samples.md",
        "Surrogates" => "surrogate.md",
        "Optimization" => "optimizations.md"
        ]
    "Tutorials" => [
        "Basics" => "tutorials.md",
        "Custom Kriging with Stheno" => "stheno.md"
        ]
    "Contributing" => "contributing.md"
    ]
)


deploydocs(
   repo = "github.com/JuliaDiffEq/Surrogates.jl.git",
)
