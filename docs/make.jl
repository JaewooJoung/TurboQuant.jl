using Documenter
using TurboQuant

DocMeta.setdocmeta!(TurboQuant, :DocTestSetup, :(using TurboQuant; using LinearAlgebra); recursive=true)

makedocs(
    sitename = "TurboQuant.jl",
    modules = [TurboQuant],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)

deploydocs(
    repo = "github.com/JaewooJoung/TurboQuant.jl.git",
    devbranch = "main",
)
