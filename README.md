# MRIRadialDelayEstimation.jl

[![CI](https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl/actions/workflows/CI.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://SebastianFlassbeck.github.io/MRIRadialDelayEstimation.jl/dev/)

Estimate and gradient delays in 2D and 3D radial MRI.

## Background

Gradient delays in MRI cause a mismatch between the assumed and actual k-space
trajectories, leading to image artefacts such as streaking and signal loss.
This package implements an iterative, data-driven algorithm that estimates
per-axis gradient delays for 2D and 3D radial acquisitions and returns a
corrected trajectory.

The algorithm works by:

1. Splitting spokes into positive / negative half-sets along each gradient axis.
2. Reconstructing each half-set on a coarse grid using conjugate-gradient (CG)
   SENSE.
3. Fitting the linear phase difference between the two half-set images to
   recover the k-space shift (delay) along that axis.
4. Iterating until convergence.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl")
```

## Quick Start

### Correct a trajectory in one call

If you have a trajectory array `trj` generated with zero delay
(shape `(3, Nr*NSpokes)` or `(3, Nr, NSpokes)` for 3D;
`(2, Nr*NSpokes)` or `(2, Nr, NSpokes)` for 2D):

```julia
using MRIRadialDelayEstimation

# 3D
trj_corrected = correct_trajectory(data, trj, img_shape; Nr=Nr)

# 2D — img_shape is a 2-tuple, e.g. (256, 256)
trj_corrected = correct_trajectory(data, trj_2d, img_shape_2d; Nr=Nr)
```

The corrected trajectory has exactly the same shape as the input.

### Estimate delays from spoke angles

```julia
# 3D — from polar (theta) and azimuthal (phi) angles
delay, delay_history = estimate_delay(data, theta, phi, Nr, img_shape)

# 2D — from azimuthal (phi) angles only
delay, delay_history = estimate_delay(data, phi, Nr, img_shape_2d)
```

### Decompose a trajectory into spoke angles

```julia
# 3D
theta, phi = decompose_kooshball(trj, Nr)

# 2D
phi = decompose_radial2d(trj, Nr)
```

## GPU Acceleration

The package has **no CUDA dependency** itself.  To run on GPU, load
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) in your script and place the
data on the GPU before calling:

```julia
using CUDA, MRIRadialDelayEstimation

# Correct trajectory with GPU-accelerated reconstructions
trj_corrected = correct_trajectory(cu(data), trj, img_shape; Nr=Nr, device=cu)

# Or estimate delays directly
delay, hist = estimate_delay(cu(data), theta, phi, Nr, img_shape; device=cu)
```

Internally computed trajectories and masks are transferred to the GPU
automatically via the `device` keyword.  Only `data` (and `cmaps`, if
provided) must be placed on the GPU by the caller.

## Multi-Coil Support

For multi-coil data, pass `data` with shape `(samples, Ncoil)`.  Coil
sensitivity maps are estimated automatically on the downsampled grid, or you
can provide pre-computed maps via the `cmaps` keyword:

```julia
delay, _ = estimate_delay(data_multicoil, theta, phi, Nr, img_shape;
    cmaps=my_coil_maps, device=cu)
```

## API Overview

| Function | Description |
|---|---|
| `correct_trajectory(data, trj, img_shape; Nr, ...)` | Estimate delays and return a corrected trajectory (2D or 3D) |
| `estimate_delay(data, theta, phi, Nr, img_shape; ...)` | Estimate per-axis gradient delays (3D) |
| `estimate_delay(data, phi, Nr, img_shape; ...)` | Estimate per-axis gradient delays (2D) |
| `decompose_kooshball(trj, Nr)` | Recover spoke angles `(theta, phi)` from a 3D kooshball trajectory |
| `decompose_radial2d(trj, Nr)` | Recover spoke angles `phi` from a 2D radial trajectory |
| `traj_2d_radial(Nr, phi; delay=(0, 0))` | Generate a 2D radial trajectory from spoke angles |


See the [documentation](https://SebastianFlassbeck.github.io/MRIRadialDelayEstimation.jl/dev/)
for full API details and keyword arguments.

## Key Parameters

| Keyword | Default | Description |
|---|---|---|
| `Niter` | `10` | Number of outer iterations (each loops over all axes) |
| `Niter_cg` | `100` | Max CG iterations per reconstruction |
| `downsample` | `(32,32,32)` 3D / `(64,64)` 2D | Coarse reconstruction grid size |
| `threshold` | `0.5` | Relative mask threshold for the phase fit |
| `converge_tol` | `1e-2/Nr` | Early-stopping tolerance on delay change |
| `device` | `identity` | Device transfer function (`identity` for CPU, `cu` for GPU) |
| `verbose` | `false` | Print per-iteration progress |

## Citation

If you use this package in your research, please cite:

> *Citation information will be added upon publication.*

## License

MIT