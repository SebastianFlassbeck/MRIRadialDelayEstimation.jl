using Test
using Random
using Statistics

using ImagePhantoms
using ImagePhantoms: phantom
using NonuniformFFTs
using MRISubspaceRecon
using MRIRadialDelayEstimation

using Unitful
using Unitful: mm

# End-to-end coverage of `correct_trajectory`, which is the only entry point
# that exercises `decompose_kooshball` / `decompose_radial2d`.  The Monte Carlo
# suites call `estimate_delay` with angles supplied directly and therefore
# never touch the decomposition code.
#
# Deliberately small (64-point matrix, single trial) so this stays cheap; the
# accuracy of the estimator itself is covered by the Monte Carlo tests.  What
# matters here is that decompose → estimate → rebuild composes correctly.
@testset "correct_trajectory (end-to-end)" begin

    Random.seed!(2468)

    T = Float32
    Tc = Complex{T}

    ############################################################
    # 3D
    ############################################################
    @testset "3D" begin
        n = 64
        fovs = (64mm, 64mm, 64mm)
        d = fovs ./ (n, n, n)
        x = (-(n ÷ 2):(n ÷ 2 - 1)) * d[1]
        y = (-(n ÷ 2):(n ÷ 2 - 1)) * d[2]
        z = (-(n ÷ 2):(n ÷ 2 - 1)) * d[3]

        params = ellipsoid_parameters(; fovs)
        scale = Tc.(rand(length(params)))
        params = [(p[1:end-1]..., scale[i]) for (i, p) in enumerate(params)]
        image0 = phantom(x, y, z, ellipsoid(params), 3)

        Nr = 2 * n
        NSpokes = 3000
        gm1, gm2 = MRISubspaceRecon.calculate_golden_means()
        theta = acos.(mod.((0:(NSpokes - 1)) * gm1, 1))
        phi   = (0:(NSpokes - 1)) * 2π * gm2
        # acos yields theta ∈ [0, π/2], so cos(theta) ≥ 0 for every spoke and
        # the z half-space split would be degenerate.  Flipping a random half
        # of the spokes to theta + π populates both z hemispheres.  Note that
        # decompose_kooshball maps such a spoke to the equivalent pair
        # (π - theta, phi + π), which regenerates the identical trajectory.
        theta[rand(length(theta)) .> 0.5] .+= π

        true_delay = T.([1.7, -2.1, 0.9] ./ Nr)

        # Nominal (zero-delay) trajectory — this is the user-facing input
        trj_nominal = T.(reshape(MRIRadialDelayEstimation.traj_3D_radial(Nr, theta, phi, zeros(T, 3)), 3, :))

        # Simulate data acquired WITH the delay
        trj_true = T.(reshape(MRIRadialDelayEstimation.traj_3D_radial(Nr, theta, phi, true_delay), 3, :))
        plan = PlanNUFFT(Tc, (n, n, n); fftshift=true)
        set_points!(plan, NonuniformFFTs._transform_point_convention.(trj_true))
        kdata = zeros(Tc, size(trj_true, 2))
        NonuniformFFTs.exec_type2!(kdata, plan, image0)
        data = Tc.(kdata .* 1e-6)

        trj_corrected = correct_trajectory(
            data, trj_nominal, (n, n, n);
            Nr, Niter=8, downsample=(32, 32, 32), converge_tol=0.0,
        )

        # Shape must be preserved exactly
        @test size(trj_corrected) == size(trj_nominal)

        # The corrected trajectory must approach the true one.  Tolerance is
        # set well below the applied delay so that a sign error or a dropped
        # correction cannot pass.
        err_corrected = maximum(abs, trj_corrected .- trj_true)
        err_nominal   = maximum(abs, trj_nominal   .- trj_true)
        @test err_corrected < 0.2 * err_nominal

        # Round-tripping a delay-free trajectory through decompose + rebuild
        th_r, ph_r = decompose_kooshball(trj_nominal, Nr)
        trj_round = reshape(traj_kooshball(Nr, reshape(th_r, :, 1), reshape(ph_r, :, 1)), size(trj_nominal))
        @test isapprox(trj_round, trj_nominal; rtol=1e-5)
    end

    ############################################################
    # 2D — this is the case the old sign pair got wrong: the
    # reflected phi fed a mirrored trajectory to the estimator.
    ############################################################
    @testset "2D" begin
        n = 64
        fovs = (64mm, 64mm)
        d = fovs ./ (n, n)
        x = (-(n ÷ 2):(n ÷ 2 - 1)) * d[1]
        y = (-(n ÷ 2):(n ÷ 2 - 1)) * d[2]

        params = ellipse_parameters(; fovs)
        scale = Tc.(rand(length(params)))
        params = [(p[1:end-1]..., scale[i]) for (i, p) in enumerate(params)]
        image0 = phantom(x, y, ellipse(params), 2)

        Nr = 2 * n
        NSpokes = 400
        τ = (sqrt(5) + 1) / 2
        phi = T.((0:(NSpokes - 1)) .* (π / τ))

        true_delay = T.([1.8, -1.3] ./ Nr)

        trj_nominal = T.(reshape(MRIRadialDelayEstimation.traj_2D_radial(Nr, reshape(phi, :, 1), zeros(T, 2)), 2, :))

        trj_true = T.(reshape(MRIRadialDelayEstimation.traj_2D_radial(Nr, reshape(phi, :, 1), true_delay), 2, :))
        plan = PlanNUFFT(Tc, (n, n); fftshift=true)
        set_points!(plan, NonuniformFFTs._transform_point_convention.(trj_true))
        kdata = zeros(Tc, size(trj_true, 2))
        NonuniformFFTs.exec_type2!(kdata, plan, image0)
        data = Tc.(kdata .* 1e-6)

        trj_corrected = correct_trajectory(
            data, trj_nominal, (n, n);
            Nr, Niter=8, downsample=(32, 32), converge_tol=0.0,
        )

        @test size(trj_corrected) == size(trj_nominal)

        err_corrected = maximum(abs, trj_corrected .- trj_true)
        err_nominal   = maximum(abs, trj_nominal   .- trj_true)
        @test err_corrected < 0.2 * err_nominal

        ph_r = decompose_radial2d(trj_nominal, Nr)
        trj_round = reshape(traj_2d_radial(Nr, reshape(ph_r, :, 1)), size(trj_nominal))
        @test isapprox(trj_round, trj_nominal; rtol=1e-5)
    end
end
