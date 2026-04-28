# MRIRadialDelayEstimation.jl

[![CI](https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl/actions/workflows/CI.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://SebastianFlassbeck.github.io/MRIRadialDelayEstimation.jl/dev/)

Estimate and correct gradient delays in 3D radial (kooshball) MRI k-space
trajectories directly from the acquired data.

## Background

Gradient delays in MRI cause a mismatch between the assumed and actual k-space
trajectories, leading to image artefacts such as streaking and signal loss.
This package implements an iterative, data-driven algorithm that estimates
per-axis gradient delays for 3D radial (kooshball) acquisitions and returns a
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

If you have a trajectory array `trj` (shape `(3, Nr*NSpokes)` or
`(3, Nr, NSpokes)`) generated with zero delay:

```julia
using MRIRadialDelayEstimation

trj_corrected = correct_trajectory(data, trj, img_shape; Nr=Nr)
```

The corrected trajectory has exactly the same shape as the input.

### Estimate delays from spoke angles

If you already have the spoke angles `theta` and `phi`:

```julia
delay, delay_history = estimate_delay(data, theta, phi, Nr, img_shape)
```

### Decompose a trajectory into spoke angles

```julia
theta, phi = decompose_kooshball(trj, Nr)
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
| `correct_trajectory(data, trj, img_shape; Nr, ...)` | Estimate delays and return a corrected trajectory |
| `estimate_delay(data, theta, phi, Nr, img_shape; ...)` | Estimate per-axis gradient delays |
| `decompose_kooshball(trj, Nr)` | Recover spoke angles `(theta, phi)` from a kooshball trajectory |
| `reconstruct_cg(data, trj, img_shape; ...)` | CG-SENSE reconstruction from non-Cartesian data |

See the [documentation](https://SebastianFlassbeck.github.io/MRIRadialDelayEstimation.jl/dev/)
for full API details and keyword arguments.

## Key Parameters

| Keyword | Default | Description |
|---|---|---|
| `Niter` | `10` | Number of outer iterations (each loops over 3 axes) |
| `Niter_cg` | `100` | Max CG iterations per reconstruction |
| `downsample` | `(32,32,32)` | Coarse reconstruction grid size |
| `threshold` | `0.5` | Relative mask threshold for the phase fit |
| `converge_tol` | `1e-2/Nr` | Early-stopping tolerance on delay change |
| `device` | `identity` | Device transfer function (`identity` for CPU, `cu` for GPU) |
| `verbose` | `false` | Print per-iteration progress |

## Citation

If you use this package in your research, please cite:

> *Citation information will be added upon publication.*

## License

MIT