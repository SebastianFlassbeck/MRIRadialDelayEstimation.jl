using Test
using Random
using MRISubspaceRecon
using MRIRadialDelayEstimation

# Wrap an angle difference into (-π, π] so comparisons are branch-cut safe.
wrap(x) = mod.(x .+ π, 2π) .- π

@testset "coordinate conventions" begin

    Random.seed!(4321)

    Nr = 64
    NSpokes = 200
    kr = collect(((-Nr + 1) / 2):((Nr - 1) / 2)) / Nr

    # Angles restricted to the ranges that the decompositions can invert
    # uniquely: theta ∈ (0, π) from acos, phi ∈ (-π, π] from atan.
    theta = π .* (0.05 .+ 0.9 .* rand(NSpokes))
    phi   = 2π .* (rand(NSpokes) .- 0.5)

    ############################################################
    # 3D: spherical coordinates
    #
    # These reference arrays are written out explicitly rather than
    # obtained from a round-trip, so they pin the actual sign
    # convention instead of merely checking self-consistency.
    ############################################################
    @testset "3D spherical convention" begin
        trj_ref = Array{Float64,3}(undef, 3, Nr, NSpokes)
        for ic in 1:NSpokes, ir in 1:Nr
            trj_ref[1, ir, ic] = sin(theta[ic]) * cos(phi[ic]) * kr[ir]
            trj_ref[2, ir, ic] = sin(theta[ic]) * sin(phi[ic]) * kr[ir]
            trj_ref[3, ir, ic] = cos(theta[ic])                * kr[ir]
        end

        # Upstream generator must agree with the spherical convention
        # (guards against a regression of the v0.9 sign flip along x).
        trj_up = reshape(traj_kooshball(Nr, reshape(theta, :, 1), reshape(phi, :, 1)), 3, Nr, NSpokes)
        @test isapprox(trj_up, trj_ref; rtol=1e-12)

        # Decomposition must recover the generating angles
        theta_est, phi_est = decompose_kooshball(trj_ref, Nr)
        @test isapprox(theta_est, theta; rtol=1e-9)
        @test maximum(abs, wrap(phi_est .- phi)) < 1e-9

        # A trajectory mirrored along x (the v0.9 convention) must NOT
        # decompose to the same angles — this is what makes the test
        # sensitive to the sign, rather than merely self-consistent.
        trj_flipped = copy(trj_ref)
        trj_flipped[1, :, :] .*= -1
        _, phi_flipped = decompose_kooshball(trj_flipped, Nr)
        @test maximum(abs, wrap(phi_flipped .- phi)) > 1e-3
    end

    ############################################################
    # 2D: polar coordinates
    ############################################################
    @testset "2D polar convention" begin
        trj2_ref = Array{Float64,3}(undef, 2, Nr, NSpokes)
        for ic in 1:NSpokes, ir in 1:Nr
            trj2_ref[1, ir, ic] = cos(phi[ic]) * kr[ir]
            trj2_ref[2, ir, ic] = sin(phi[ic]) * kr[ir]
        end

        trj2 = reshape(traj_2d_radial(Nr, reshape(phi, :, 1)), 2, Nr, NSpokes)
        @test isapprox(trj2, trj2_ref; rtol=1e-12)

        # Decomposition must recover the generating angles
        phi_est = decompose_radial2d(trj2_ref, Nr)
        @test maximum(abs, wrap(phi_est .- phi)) < 1e-9

        # Sensitivity to the x sign, as above
        trj2_flipped = copy(trj2_ref)
        trj2_flipped[1, :, :] .*= -1
        phi_flipped = decompose_radial2d(trj2_flipped, Nr)
        @test maximum(abs, wrap(phi_flipped .- phi)) > 1e-3
    end

    ############################################################
    # 2D/3D consistency: traj_2d_radial must equal the first two
    # components of traj_kooshball at theta = π/2, including delays.
    # `estimate_delay` builds its trajectories via traj_kooshball
    # while `correct_trajectory` returns traj_2d_radial, so a
    # mismatch here would corrupt the 2D delay estimate.
    ############################################################
    @testset "2D/3D generator consistency" begin
        theta_eq = fill(π / 2, NSpokes, 1)
        phi_col  = reshape(phi, :, 1)

        for delay in ([0.0, 0.0], [0.13, -0.07])
            k3 = reshape(traj_kooshball(Nr, theta_eq, phi_col; delay=[delay[1], delay[2], 0.0]), 3, Nr, NSpokes)
            k2 = reshape(traj_2d_radial(Nr, phi_col; delay=delay), 2, Nr, NSpokes)
            @test isapprox(k2, k3[1:2, :, :]; rtol=1e-12)
        end
    end
end
