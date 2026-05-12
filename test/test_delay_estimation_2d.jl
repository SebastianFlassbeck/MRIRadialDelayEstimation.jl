using Test
using LinearAlgebra
using Random
using Statistics

using ImagePhantoms
using ImagePhantoms: phantom
using NonuniformFFTs
using MRISubspaceRecon

using Unitful
using Unitful: mm

@testset "delay estimation 2D (Monte Carlo)" begin

    Random.seed!(1234)

    ############################
    # Imaging parameters
    ############################
    T = Float32
    Tc = Complex{T}

    fovs = (256mm, 256mm)
    nx, ny = (256, 256)
    dx, dy = fovs ./ (nx, ny)
    NSpokes = 500
    x = (-(nx÷2):(nx÷2-1)) * dx
    y = (-(ny÷2):(ny÷2-1)) * dy

    ############################
    # Generate 2D phantom
    ############################
    params = ellipse_parameters(; fovs)
    scale = Tc.(rand(length(params)))
    params = [(p[1:end-1]..., scale[i]) for (i, p) in enumerate(params)]
    ob = ellipse(params)
    image0 = phantom(x, y, ob, 2)

    ############################
    # Generate 2D radial spoke angles (golden angle)
    ############################
    Nr = 2 * nx
    τ = (sqrt(5) + 1) / 2
    angle_GR = T(π / τ)
    phi = (0:(NSpokes-1)) .* angle_GR

    ############################
    # Test parameters
    ############################
    SNR       = 20.0
    Ntrials   = 5
    Niter     = 20
    max_delay = 2.5 / Nr
    threshold = 0.5
    downsample = (32, 32)
    img_shape  = ntuple(_ -> maximum((nx, ny)), 2)
    error_tol  = 1e-4

    signal_amplitude = maximum(abs.(image0))
    noise_sigma = sqrt(signal_amplitude / SNR * nx * ny)

    nufft_plan = PlanNUFFT(Tc, (nx, ny); fftshift=true)

    ############################
    # Monte Carlo loop
    ############################
    errors_all = zeros(T, 2, Ntrials)

    for itrial in 1:Ntrials
        # Random true delay
        true_delay = T.(max_delay .* (2 .* rand(2) .- 1))

        # Simulate k-space data with the true delay
        trj_true = T.(reshape(traj_2D_radial(Nr, reshape(phi, :, 1), true_delay), 2, :))
        set_points!(nufft_plan, NonuniformFFTs._transform_point_convention.(trj_true))

        kdata = zeros(Tc, size(trj_true, 2))
        NonuniformFFTs.exec_type2!(kdata, nufft_plan, image0)

        # Add noise
        noise = Tc(noise_sigma / sqrt(2)) .* randn(Tc, length(kdata))
        kdata .+= noise

        # Scale data
        data = Tc.(kdata .* 1e-6)

        # Estimate delay
        delay, _ = estimate_delay(
            data, phi, Nr, (nx, ny);
            Niter, threshold, downsample, converge_tol=0.0,
        )

        errors_all[:, itrial] .= delay .- true_delay
    end

    rmse = sqrt.(mean(errors_all .^ 2; dims=2))

    @testset "RMSE per axis below tolerance" begin
        for (idir, label) in enumerate(["x", "y"])
            @test rmse[idir] < error_tol
        end
    end
end