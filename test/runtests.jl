using Test

@testset "MRIRadialDelayEstimation.jl" begin
    # Fast, deterministic checks first so convention regressions surface
    # before the expensive Monte Carlo suites run.
    include("test_conventions.jl")
    include("test_decompose_kooshball.jl")
    include("test_correct_trajectory.jl")
    include("test_delay_estimation.jl")
    include("test_delay_estimation_2d.jl")
end