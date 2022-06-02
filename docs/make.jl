using Documenter, Surrogates

include("pages.jl")

makedocs(
    sitename="Surrogates.jl",
    format = Documenter.HTML(analytics = "UA-90474609-3",
                         assets = ["assets/favicon.ico"],
                         canonical="https://surrogates.sciml.ai/stable/"),
    pages = pages
)


deploydocs(
   repo = "github.com/SciML/Surrogates.jl.git",
)
