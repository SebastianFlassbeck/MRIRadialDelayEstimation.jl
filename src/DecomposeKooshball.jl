"""
    decompose_kooshball(trj, Nr) -> (theta, phi, kr)

Recover the spoke angles `theta`, `phi` and readout positions `kr` from a
3D radial (kooshball) trajectory that was generated with `delay = [0, 0, 0]`.

The trajectory is assumed to follow the spherical-coordinate convention used by
`MRISubspaceRecon.traj_kooshball` (v0.10 and later):
```
k[1, ir, ic] = sin(θ) * cos(φ) * kr[ir]
k[2, ir, ic] = sin(θ) * sin(φ) * kr[ir]
k[3, ir, ic] = cos(θ)          * kr[ir]
```

# Arguments
- `trj`: trajectory array, either `(3, Nr, NSpokes)` or `(3, Nr * NSpokes)`.
  In the latter case `Nr` is used to reshape.
- `Nr`: number of readout points per spoke.

# Returns
- `theta::Vector`: polar angle per spoke (length `NSpokes`), in `[0, π]`.
- `phi::Vector`: azimuthal angle per spoke (length `NSpokes`), in `(-π, π]`.
"""
function decompose_kooshball(trj::AbstractArray{<:Real}, Nr::Integer)
    # Reshape to (3, Nr, NSpokes) if needed
    if ndims(trj) == 2
        @assert size(trj, 1) == 3
        NSpokes = size(trj, 2) ÷ Nr
        trj = reshape(trj, 3, Nr, NSpokes)
    else
        @assert ndims(trj) == 3 && size(trj, 1) == 3
        NSpokes = size(trj, 3)
    end

    # Nominal kr positions (same for every spoke)
    kr = collect(eltype(trj), ((-Nr + 1) / 2):((Nr - 1) / 2)) / Nr

    # For each spoke, k[:,ir,ic] = [a; b; c] * kr[ir], where
    #   a = sin(θ)cos(φ),  b = sin(θ)sin(φ),  c = cos(θ)
    # Recover the slopes via least-squares fit against kr.
    theta = Vector{eltype(trj)}(undef, NSpokes)
    phi   = Vector{eltype(trj)}(undef, NSpokes)

    for ic in 1:NSpokes
        a = kr \ @view(trj[1, :, ic])   #  sin(θ)cos(φ)
        b = kr \ @view(trj[2, :, ic])   #  sin(θ)sin(φ)
        c = kr \ @view(trj[3, :, ic])   #  cos(θ)

        # θ from cos(θ)
        cos_theta = clamp(c, -one(eltype(trj)), one(eltype(trj)))
        theta[ic] = acos(cos_theta)

        # φ from atan2(sin(θ)sin(φ), sin(θ)cos(φ)) = atan2(b, a)
        phi[ic] = atan(b, a)

    end

    return theta, phi
end
