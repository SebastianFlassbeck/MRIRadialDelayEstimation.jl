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
        canonical="https://flasss01.github.io/MRIRadialDelayEstimation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/flasss01/MRIRadialDelayEstimation.jl",
    devbranch="main",
)