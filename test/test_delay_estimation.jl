using Test
using LinearAlgebra
using Random
using Statistics

using ImagePhantoms
using ImagePhantoms: phantom
using NonuniformFFTs
using MRISubspaceRecon
using MRIRadialDelayEstimation

using Unitful
using Unitful: mm

@testset "delay estimation (Monte Carlo)" begin

    Random.seed!(42)

    ############################
    # Imaging parameters
    ############################
    T = Float32
    Tc = Complex{T}

    fovs = (256mm, 256mm, 256mm)
    nx, ny, nz = (256, 256, 256)
    dx, dy, dz = fovs ./ (nx, ny, nz)
    NSpokes = 10000
    x = (-(nx÷2):(nx÷2-1)) * dx
    y = (-(ny÷2):(ny÷2-1)) * dy
    z = (-(nz÷2):(nz÷2-1)) * dz

    ############################
    # Generate phantom
    ############################
    params = ellipsoid_parameters(; fovs)
    scale = Tc.(rand(length(params)))
    params = [(p[1:end-1]..., scale[i]) for (i, p) in enumerate(params)]
    ob = ellipsoid(params)
    image0 = phantom(x, y, z, ob, 3)

    ############################
    # Generate kooshball spoke angles
    ############################
    Nr = 2 * nx
    gm1, gm2 = MRISubspaceRecon.calculate_golden_means()
    theta = acos.(mod.((0:(NSpokes-1)) * gm1, 1))
    phi   = (0:(NSpokes-1)) * 2π * gm2
    theta[rand(length(theta)) .> 0.5] .+= π

    ############################
    # Test parameters
    ############################
    SNR       = 20.0
    Ntrials   = 5
    Niter     = 20
    max_delay = 2.5 / Nr
    threshold = 0.5
    downsample = (32, 32, 32)
    img_shape  = ntuple(_ -> maximum((nx, ny, nz)), 3)
    error_tol  = 1e-4

    signal_amplitude = maximum(abs.(image0))
    noise_sigma = sqrt(signal_amplitude / SNR * nx * ny * nz)

    nufft_plan = PlanNUFFT(Tc, (nx, ny, nz); fftshift=true)

    ############################
    # Monte Carlo loop
    ############################
    errors_all = zeros(T, 3, Ntrials)

    for itrial in 1:Ntrials
        # Random true delay
        true_delay = T.(max_delay .* (2 .* rand(3) .- 1))

        # Simulate k-space data with the true delay
        trj_true = T.(reshape(traj_kooshball(Nr, reshape(theta,:,1), reshape(phi,:,1); delay=true_delay), 3, :))
        set_points!(nufft_plan, NonuniformFFTs._transform_point_convention.(trj_true))

        kdata = zeros(Tc, size(trj_true, 2))
        NonuniformFFTs.exec_type2!(kdata, nufft_plan, image0)

        # Add noise
        noise = Tc(noise_sigma / sqrt(2)) .* randn(Tc, length(kdata))
        kdata .+= noise

        # Scale data
        data = Tc.(reshape(kdata, :, 1, 1) .* T(1e-6))

        # Estimate delay
        delay, _ = estimate_delay(
            data, theta, phi, Nr, (nx, ny, nz);
            Niter, threshold, downsample, converge_tol=0.0,
        )

        errors_all[:, itrial] .= delay .- true_delay
    end

    rmse = sqrt.(mean(errors_all .^ 2; dims=2))

    @testset "RMSE per axis below tolerance" begin
        for (idir, label) in enumerate(["x", "y", "z"])
            @test rmse[idir] < error_tol
        end
    end
end