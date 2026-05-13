using Test

@testset "MRIRadialDelayEstimation.jl" begin
    include("test_decompose_kooshball.jl")
    include("test_delay_estimation.jl")
    include("test_delay_estimation_2d.jl")
end