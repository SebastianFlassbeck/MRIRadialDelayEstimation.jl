# ── Device transfer helpers ──────────────────────────────────────────────
"""
    to_device(device, x)

Transfer `x` to the given device.  `device` should be a callable such as
`cu` (from CUDA.jl) or `identity` / `cpu` for CPU arrays.
"""
to_device(device, x::AbstractArray) = device(x)
to_device(::typeof(identity), x::AbstractArray) = x
to_device(device, x::Nothing) = nothing
to_device(device, xs::Vector{<:AbstractArray}) = [to_device(device, xi) for xi in xs]

"""
    to_cpu(x)

Collect a (potentially GPU-backed) array to a plain CPU `Array`.
No-op for arrays that are already `Array`.
"""
to_cpu(x::Array) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Nothing) = nothing

# ── CG reconstruction ───────────────────────────────────────────────────
"""
    reconstruct_cg(data, trj, img_shape;
        cmaps=(1,), sample_mask=trues(size(trj, 2)),
        Niter_cg=100, verbose=false,
    ) -> x

CG reconstruction from non-Cartesian k-space data.

Builds the Toeplitz-based NFFT normal operator via
`MRISubspaceRecon.NFFTNormalOp` and solves `AᴴA x = Aᴴ data` with
conjugate gradients.

All input arrays (`data`, `trj`, `cmaps`) should already reside on the
desired device (CPU or GPU) before calling this function.  The caller
(e.g. `estimate_delay`) is responsible for device transfers.

# Arguments
- `data`: k-space data, shape `(samples,)` for single coil or
  `(samples, Ncoil)` for multi-coil.
- `trj`: trajectory, shape `(Ndim, samples)` or `(Ndim, samples, 1)`.
  A trailing subspace/time-frame dimension is added automatically.
- `img_shape::NTuple{N,Int}`: reconstruction matrix size.

# Keyword Arguments
- `cmaps`: coil sensitivity maps.  Pass `(1,)` (default) for a single-coil
  reconstruction.  Pass a `Vector`/`Tuple` of arrays (one per coil, each
  of size `img_shape`) for a SENSE-combined multi-coil reconstruction.
- `sample_mask::AbstractVector{Bool}`: which k-space samples to include.
  Default `trues(size(trj, 2))`.
- `Niter_cg::Int`: maximum CG iterations.  Default `100`.
- `verbose::Bool`: print timing info.  Default `false`.

# Returns
- `x::Array{Complex,N}`: reconstructed image of size `(img_shape...)`.
"""
function reconstruct_cg(
    data::AbstractArray{Tc},
    trj::AbstractArray{T},
    img_shape::NTuple{N,<:Integer};
    cmaps=(1,),
    sample_mask::AbstractVector{Bool} = trues(size(trj, 2)),
    Niter_cg::Integer = 100,
    verbose::Bool = false,
) where {T <: Real, Tc <: Complex{T}, N}

    # Add subspace/time-frame dimension for MRISubspaceRecon
    trj = ndims(trj) == 2 ? reshape(trj, size(trj, 1), :, 1) : trj
    sample_mask = reshape(sample_mask, :, 1)
    U_cpu = T.([1.0])
    U = similar(trj, T, size(U_cpu)...)
    copyto!(U, U_cpu)
    Ncoil = length(cmaps)

    data = reshape(data, :, 1, Ncoil)

    AᴴA = NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask, verbose)

    if length(cmaps) == 1
        xbp = calculate_backprojection(data, trj, img_shape;
            U, sample_mask, verbose)
    else
        xbp = calculate_backprojection(data, trj, cmaps;
            U, sample_mask, verbose)
    end


    x = similar(xbp, Tc, size(xbp)...)
    fill!(x, zero(Tc))
    x = vec(x)
    bi = vec(xbp)

    cg!(x, AᴴA, bi; maxiter=Niter_cg, verbose, reltol=0)
    return reshape(x,img_shape...)

end


function traj_3D_radial(Nr, theta, phi, delay)
    theta = reshape(theta, :, 1)
    phi   = reshape(phi,   :, 1)
    trj = traj_kooshball(Nr, theta, phi; delay)
    return trj
end

function traj_2D_radial(Nr, phi, _delay)
    phi   = reshape(phi,   :, 1)
    theta = similar(phi)
    theta .= π/2
    trj = traj_kooshball(Nr, theta, phi; delay=[_delay[1],_delay[2], 0])
    return trj[1:2, ntuple(_ -> :, ndims(trj) - 1)...]
end