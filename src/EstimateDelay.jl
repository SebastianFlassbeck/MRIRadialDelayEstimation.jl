"""
    estimate_delay(data, theta, phi, Nr, img_shape; kwargs...) -> (delay, delay_history)

Estimate gradient delay from 3D radial (kooshball) k-space data by iteratively
splitting spokes into positive/negative half-sets along each axis, reconstructing
each half on a coarse grid using CG, and fitting the resulting linear phase difference.

# Arguments
- `data`: k-space data, already on the target device (CPU or GPU).
  Accepted shapes:
  - `(samples,)` — single coil
  - `(samples, Ncoil)` — multi-coil
- `theta::AbstractVector`: polar angles of the spokes (length `NSpokes`).
- `phi::AbstractVector`: azimuthal angles of the spokes (length `NSpokes`).
- `Nr::Integer`: number of readout points per spoke.
- `img_shape::NTuple{3,Integer}`: reconstruction matrix size, e.g. `(256, 256, 256)`.

# Keyword Arguments
- `delay_init`: initial delay estimate, length-3 vector. Default `zeros(T, 3)`
  where `T = real(eltype(data))`.
- `Niter`: number of outer iterations (each iteration loops over 3 axes).
  Default `10`.
- `Niter_cg`: maximum CG iterations per reconstruction. Default `100`.
- `threshold`: relative threshold (fraction of the 90th percentile) used to
  build the spatial mask for the linear fit. Default `0.5`.
- `converge_tol`: early-stopping tolerance on the maximum absolute delay
  change per iteration.  Default `1e-2 / Nr`.
- `downsample`: coarse reconstruction grid size. Default `(32, 32, 32)`.
- `cmaps`: coil sensitivity maps.  Pass `nothing` (default) to auto-estimate
  them via `calculate_coil_maps` when multi-coil data is detected.  Pass a
  `Vector` of arrays (one per coil, each of size `downsample`) to use
  pre-computed maps.  Must already reside on the same device as `data`.
  Ignored for single-coil data.
- `device`: device transfer function, e.g. `cu` (from CUDA.jl) for GPU or
  `identity` (default) for CPU.  Used only to transfer internally computed
  trajectories and masks to the target device.  The caller is responsible
  for placing `data` (and `cmaps`, if provided) on the device beforehand.
- `verbose::Bool`: print per-iteration delay estimates. Default `false`.

# Returns
- `delay::Vector{T}`: estimated delay per axis (length 3), in the same units
  as the trajectory produced by `traj_kooshball`.
- `delay_history::Matrix{T}`: delay estimates at each iteration, size
  `(3, Niter)`.

# Example
```julia
using CUDA, MRIRadialDelayEstimation

# CPU
delay, hist = estimate_delay(data, theta, phi, Nr, img_shape)

# GPU — transfer data before calling
delay, hist = estimate_delay(cu(data), theta, phi, Nr, img_shape; device=cu)
```
"""
function estimate_delay(
    data::AbstractArray{<:Complex},
    theta::AbstractVector,
    phi::AbstractVector,
    Nr::Integer,
    img_shape::NTuple{3,<:Integer};
    delay_init::AbstractVector{<:Real} = zeros(real(eltype(data)), 3),
    Niter::Integer = 10,
    Niter_cg::Integer = 100,
    threshold::Real = 0.5,
    converge_tol = 1e-2 / Nr,
    downsample::NTuple{3,<:Integer} = (32, 32, 32),
    cmaps = nothing,
    device = identity,
    verbose::Bool = false
)
    T = real(eltype(data))

    # Make img_shape isotropic
    img_shape_iso = ntuple(_ -> maximum(img_shape), 3)
    # Determine single vs multi-coil
    is_multicoil = ndims(data) == 2 && size(data, 2) > 1
    Ncoil = is_multicoil ? size(data, 2) : 1
 
    if !is_multicoil && ndims(data) == 2
        data = vec(data)  # single coil: flatten to (samples,)
    end

    # traj_kooshball expects column matrices (NSpokes, 1)
    theta = reshape(vec(theta), :, 1)
    phi   = reshape(vec(phi),   :, 1)

    # Estimate coil maps on the downsampled grid if multi-coil and not provided
    if is_multicoil && cmaps === nothing
        trj = traj_kooshball(Nr, theta, phi; delay=delay_init)
        trj = reshape(trj, 3, :, 1)
        trj = reshape(trj .* T.(img_shape_iso ./ downsample), 3, :, 1)
        sample_mask = reshape(vec(all(abs.(trj) .< 0.5; dims=1)), :, 1)
        cmaps = calculate_coil_maps(reshape(data,:,1,Ncoil), device(trj), downsample; sample_mask=device(BitMatrix(sample_mask)), verbose)

        verbose && println("  Estimated coil maps ($(length(cmaps)) coils)")
    end

    # For single-coil, ensure cmaps is nothing
    if !is_multicoil
        cmaps=(1,)
    end

    # Initialise delay and history
    delay = T.(copy(delay_init))
    delay_history = zeros(T, 3, Niter)

    for iq in 1:Niter
        delay_prev = copy(delay)

        for idir in 1:3
            # Build trajectory with current delay estimate (CPU — cheap)
            trj = T.(traj_kooshball(Nr, theta, phi; delay))
            trj = reshape(trj, 3, Nr, :)

            # Split spokes based on trajectory direction along current axis
            split_trj = trj[idir, 1, :] .> 0

            # Scale trajectory for downsampled grid and compute calibration mask
            trj_calib = reshape(trj .* T.(img_shape_iso ./ downsample), 3, :)
            mask_calib = vec(all(abs.(trj_calib) .< 0.5; dims=1))

            split_trj_rep = vec(repeat(split_trj, inner=Nr))

            # Transfer trajectory to device for CG recon
            trj_calib = device(trj_calib)

            # Downsampled CG recon for positive-half spokes
            recon_phalf = reconstruct_cg(
                data, trj_calib, downsample;
                cmaps, sample_mask=device(mask_calib .& split_trj_rep), Niter_cg
            )

            # Downsampled CG recon for negative-half spokes
            recon_nhalf = reconstruct_cg(
                data, trj_calib, downsample;
                cmaps, sample_mask=device(mask_calib .& (.!split_trj_rep)), Niter_cg
            )

            # Bring reconstructions to CPU for phase fitting
            recon_phalf = to_cpu(recon_phalf)
            recon_nhalf = to_cpu(recon_nhalf)

            # Phase difference between the two half-sets
            pd = angle.(recon_phalf ./ recon_nhalf)

            # Build mask on the downsampled grid
            combined = recon_phalf .+ recon_nhalf
            mask_ds = abs.(combined) .> threshold * quantile(vec(abs.(combined)), 0.9)
            mask_ds = opening(mask_ds, strel_box(mask_ds, r=1))
            
            (Xds, Yds, Zds) = ndgrid(
                range(-0.5, 0.5, downsample[1]),
                range(-0.5, 0.5, downsample[2]),
                range(-0.5, 0.5, downsample[3]),
            )

               Pos_ds = cat(dims=2,
                ones(eltype(Xds), length(Xds[mask_ds])),
                Xds[mask_ds],
                Yds[mask_ds],
                Zds[mask_ds],
            )

            # Linear least-squares fit: phase = [1, x, y, z] * p
            p_fit = Pos_ds \ pd[mask_ds]

            # Update delay along the current axis
            Δk_cart = p_fit[idir + 1]/(4π) # 2π from Fourier transform and factor 2 because we need to shift each hemisphere by half
            Δk_rad = Δk_cart/img_shape_iso[1]
            delay[idir] += Δk_rad
            delay_history[idir, iq] = delay[idir]

            if verbose
                println("  iter=$iq, axis=$idir: delay = $delay")
            end
        end

        # Early stopping: check if all axes converged
        if maximum(abs.(delay .- delay_prev)) < converge_tol
            verbose && println("  Converged at iteration $iq (Δdelay < $(converge_tol))")
            # Fill remaining history with final values
            for iq_remaining in (iq+1):Niter
                delay_history[:, iq_remaining] = delay
            end
            break
        end
    end

    return delay, delay_history
end

"""
    correct_trajectory(data, trj, img_shape; Nr, kwargs...) -> trj_corrected

Estimate and correct gradient delays for a 3D radial (kooshball) trajectory.

This is a convenience wrapper around [`estimate_delay`](@ref) that:
1. Decomposes `trj` into spoke angles via [`decompose_kooshball`](@ref).
2. Estimates the gradient delay.
3. Returns a corrected trajectory with the estimated delay applied.

The input trajectory must have been generated with `delay = [0, 0, 0]`;
angles are recovered under that assumption.

`data` must already reside on the target device (CPU or GPU).  All keyword
arguments (except `Nr`) are forwarded to [`estimate_delay`](@ref).

# Arguments
- `data`: k-space data, already on the target device.
- `trj`: trajectory array of arbitrary shape, provided the first dimension
  is 3 (spatial axes) and the total number of elements is `3 * Nr * NSpokes`.
  Common shapes include `(3, Nr*NSpokes)`, `(3, Nr, NSpokes)`, or higher-dimensional
  layouts.  The returned trajectory will have exactly the same shape.
- `img_shape::NTuple{3,Integer}`: reconstruction matrix size.

# Keyword Arguments
- `Nr::Integer`: number of readout points per spoke.  Required when `trj` is
  `(3, Nr*NSpokes)`; inferred from `size(trj, 2)` when `trj` is 3D.
- All other kwargs are forwarded to `estimate_delay`.

# Returns
- `trj_corrected`: trajectory with the estimated delay applied.  The shape
  matches the input `trj` exactly, regardless of its original dimensionality.

# Example
```julia
using CUDA, MRIRadialDelayEstimation

trj_corrected = correct_trajectory(cu(data), trj, img_shape; Nr=Nr, device=cu)
```
"""
function correct_trajectory(
    data::AbstractArray{<:Complex},
    trj::AbstractArray{<:Real},
    img_shape::NTuple{3,<:Integer};
    Nr::Integer = size(trj, 2),
    kwargs...,
)
    original_shape = size(trj)
    theta, phi = decompose_kooshball(trj, Nr)
    delay, _ = estimate_delay(data, theta, phi, Nr, img_shape; kwargs...)
    # traj_kooshball expects column matrices (NSpokes, 1)
    trj_corrected = traj_kooshball(Nr, reshape(theta,:,1), reshape(phi,:,1); delay)
    return reshape(trj_corrected, original_shape)
end
