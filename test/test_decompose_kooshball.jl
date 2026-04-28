using Test
using LinearAlgebra
using Random
using MRISubspaceRecon
using MRIRadialDelayEstimation

Random.seed!(0)

@testset "decompose_kooshball" begin

    Nr = 512
    NSpokes = 1000

    ############################
    # Generate random angles
    ############################
    theta_true = π .* rand(NSpokes,1)          
    phi_true   = 2π .* rand(NSpokes,1)          

    ############################
    # Generate nominal trajectory 
    ############################
    trj = reshape(traj_kooshball(Nr, theta_true, phi_true),3,Nr,:)

    ############################
    # Decompose
    ############################
    theta_est, phi_est = decompose_kooshball(trj, Nr)

    ############################
    # Verify by reconstructing the trajectory
    ############################
    trj_rebuilt = reshape(traj_kooshball(Nr, reshape(theta_est,:,1), reshape(phi_est,:,1)),3,Nr,:)

    @testset "test trajectory" begin
        @test isapprox(trj_rebuilt, trj; atol=1e-5)
    end


end