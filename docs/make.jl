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
    "Tutorials" => "tutorials.md"
    "Contributing" => "contributing.md"
    ]
)


deploydocs(
   repo = "github.com/JuliaDiffEq/Surrogates.jl.git",
)
