# Architecture: Plenoxels

## Motivation

NeRF requires a neural network, making training slow. Plenoxels shows that the neural network is not necessary — a sparse voxel grid with spherical harmonics can represent the same scene.

## Core Idea

Store density and spherical harmonic coefficients in a sparse 3D voxel grid. Optimize the grid directly via gradient descent. No neural network is needed — the grid IS the representation.

## Architecture

### Overview

![plenoxels architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Coordinate (x,y,z) | `input` | shape: [3] |
| 2 | Sparse Voxel Grid | `embedding` | stores density + SH coefficients per voxel |
| 3 | Trilinear Interp | `identity` | interpolate between voxel corners |
| 4 | Spherical Harmonics | `linear` | degree 2 SH, 9 coeffs -> RGB |
| 5 | Density + Color | `output` |  |

</details>

### Components

1. **Sparse voxel grid** — Only stores voxels near the object surface (non-empty). Pruning removes empty voxels. 2. **Spherical harmonics** — Degree-2 SH (9 coefficients per voxel) for view-dependent color. 3. **Trilinear interpolation** — Interpolate density and SH coefficients from the 8 nearest voxels. 4. **Direct optimization** — No neural network; the voxel grid is optimized directly via gradient descent. 5. **Coarse-to-fine** — Start at low resolution, progressively increase resolution.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Store density and spherical harmonic coefficients in a sparse 3D voxel grid. Optimize the grid directly via gradient descent. No neural network is needed — the grid IS the representation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NeRF, Instant-NGP. Successor: K-Planes, triplane representations.

## References

- Fridovich-Keil et al. 2022
