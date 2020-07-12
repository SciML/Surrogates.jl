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
        "RandomForestSurrogate" => "randomforest.md",
        "Radials" => "radials.md",
        "Kriging" => "kriging.md",
        "Lobachesky" => "lobachesky.md",
        "LinearSurrogate" => "LinearSurrogate.md",
        "InverseDistance" => "InverseDistance.md"
        ]
    "Benchmarks" => [
        "Sphere function" => "sphere_function.md"
    ]
    "Contributing" => "contributing.md"
    ]
)


deploydocs(
   repo = "github.com/SciML/Surrogates.jl.git",
)
