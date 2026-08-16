"""
    traj_2d_radial(Nr, phi; delay=(0, 0)) -> k

Generate a 2D radial trajectory from spoke angles `phi`.

The trajectory follows the standard polar-coordinate convention, which is the
2D case of `MRISubspaceRecon.traj_kooshball` at `theta = π/2`:
```
k[1, ir, ic] = cos(φ) * (kr[ir] + delay[1])
k[2, ir, ic] = sin(φ) * (kr[ir] + delay[2])
```

This is equivalent to calling `traj_kooshball` with `theta = π/2` and
taking the first 2 rows, but avoids the 3D overhead.

# Arguments
- `Nr::Integer`: number of readout points per spoke.
- `phi`: spoke angles, shape `(NSpokes, Nt)` or `(NSpokes, 1)`.

# Keyword Arguments
- `delay`: gradient delay per axis, length-2 vector or tuple. Default `(0, 0)`.

# Returns
- `k::Array{T,3}`: trajectory of shape `(2, Nr * NSpokes, Nt)`.
"""
function traj_2d_radial(Nr::Integer, phi::AbstractMatrix; delay=(0, 0))
    Ncyc, Nt = size(phi)
    T_el = eltype(phi)

    kr = collect(T_el, ((-Nr + 1) / 2):((Nr - 1) / 2)) / Nr

    cphi = cos.(phi)
    sphi = sin.(phi)

    k = Array{T_el,3}(undef, 2, Nr * Ncyc, Nt)

    for it in axes(k, 3)
        ki = Array{T_el,3}(undef, 2, Nr, Ncyc)
        Threads.@threads for ic in 1:Ncyc
            for ir in 1:Nr
                ki[1, ir, ic] = cphi[ic, it] * (kr[ir] + delay[1])
                ki[2, ir, ic] = sphi[ic, it] * (kr[ir] + delay[2])
            end
        end
        k[:, :, it] = reshape(ki, 2, :)
        @. k[:, :, it] = max(min(k[:, :, it], T_el(0.5)), T_el(-0.5))
    end

    return k
end

"""
    decompose_radial2d(trj, Nr) -> phi

Recover the spoke angles `phi` from a 2D radial trajectory that was
generated with `delay = [0, 0]`.

The trajectory is assumed to follow the polar-coordinate convention:
```
k[1, ir, ic] = cos(φ) * kr[ir]
k[2, ir, ic] = sin(φ) * kr[ir]
```

# Arguments
- `trj`: trajectory array, either `(2, Nr, NSpokes)` or `(2, Nr * NSpokes)`.
  In the latter case `Nr` is used to reshape.
- `Nr`: number of readout points per spoke.

# Returns
- `phi::Vector`: azimuthal angle per spoke (length `NSpokes`), in `(-π, π]`.
"""
function decompose_radial2d(trj::AbstractArray{<:Real}, Nr::Integer)
    # Reshape to (2, Nr, NSpokes) if needed
    if ndims(trj) == 2
        @assert size(trj, 1) == 2 "First dimension of 2D trajectory must be 2, got $(size(trj, 1))"
        NSpokes = size(trj, 2) ÷ Nr
        trj = reshape(trj, 2, Nr, NSpokes)
    else
        @assert ndims(trj) == 3 && size(trj, 1) == 2 "Expected (2, Nr, NSpokes) array"
        NSpokes = size(trj, 3)
    end

    # Nominal kr positions (same for every spoke)
    kr = collect(eltype(trj), ((-Nr + 1) / 2):((Nr - 1) / 2)) / Nr

    phi = Vector{eltype(trj)}(undef, NSpokes)

    for ic in 1:NSpokes
        # Fit slopes: k[1,:,ic] = cos(φ) * kr, k[2,:,ic] = sin(φ) * kr
        a = kr \ @view(trj[1, :, ic])   #  cos(φ)
        b = kr \ @view(trj[2, :, ic])   #  sin(φ)

        # φ from atan2(sin(φ), cos(φ)) = atan2(b, a)
        phi[ic] = atan(b, a)
    end

    return phi
end