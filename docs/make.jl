using Documenter, Surrogates
makedocs(
    sitename="Surrogates.jl",
    pages = [
    "index.md"
    "User guide" => [
        "Samples" => "samples.md",
        "Surrogates" => "surrogates.md",
        "Optimization" => "optimizations.md"
        ]
    "Tutorials" => "tutorials.md"
    "Contributing" => "contributing.md"
    ]
)
