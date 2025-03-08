using Documenter, Surrogates

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = "Documenter"
using Plots

include("pages.jl")

makedocs(sitename = "Surrogates.jl",
    linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(analytics = "UA-90474609-3",
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Surrogates/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/Surrogates.jl.git")
