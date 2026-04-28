# MRIRadialDelayEstimation.jl

Estimate and correct gradient delays in 3D radial (kooshball) MRI k-space trajectories.

## Overview

Gradient delays in MRI cause misalignment between the assumed and actual k-space
trajectories, leading to image artefacts.  This package estimates per-axis delays
from the acquired k-space data itself and produces a corrected trajectory.

The algorithm works by:
1. Splitting spokes into positive/negative half-sets along each gradient axis.
2. Reconstructing each half-set on a coarse grid using conjugate-gradient SENSE.
3. Fitting the linear phase difference between half-sets to recover the delay.
4. Iterating until convergence.

## Quick Start

```julia
using MRIRadialDelayEstimation

# If you have spoke angles (theta, phi):
delay, delay_history = estimate_delay(data, theta, phi, Nr, img_shape)

# If you have a trajectory array directly:
trj_corrected = correct_trajectory(data, trj, img_shape; Nr=Nr)
```

### GPU acceleration

The package itself has no CUDA dependency.  To run on GPU, load CUDA.jl in your
script and pass the data on the GPU:

```julia
using CUDA, MRIRadialDelayEstimation

delay, _ = estimate_delay(cu(data), theta, phi, Nr, img_shape; device=cu)

# or
trj_corrected = correct_trajectory(cu(data), trj, img_shape; Nr=Nr, device=cu)
```

## Installation

```julia
using Pkg
Pkg.add("MRIRadialDelayEstimation")
```

Or for the development version:

```julia
Pkg.add(url="https://github.com/SebastianFlassbeck/MRIRadialDelayEstimation.jl")
```