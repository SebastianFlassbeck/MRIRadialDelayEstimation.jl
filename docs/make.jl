using Documenter
using MRIRadialDelayEstimation

DocMeta.setdocmeta!(
    MRIRadialDelayEstimation,
    :DocTestSetup,
    :(using MRIRadialDelayEstimation);
    recursive=true,
)

makedocs(;
    modules=[MRIRadialDelayEstimation],
    authors="MRIRadialDelayEstimation contributors",
    sitename="MRIRadialDelayEstimation.jl",
    format=Documenter.HTML(;
        canonical="https://SebastianFlassbeck.github.io/MRIRadialDelayEstimation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Monte Carlo Validation" => "examples/monte_carlo_validation.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl",
    devbranch="master",
)