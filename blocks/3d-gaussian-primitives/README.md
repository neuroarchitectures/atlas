# 3D Gaussian Primitives

## Design Philosophy

Represent a scene as millions of explicit 3D Gaussians — position, opacity, anisotropic 3×3 covariance, and view-dependent color via spherical harmonics — initialized from the SfM point cloud. Explicit primitives give editable, rasterizable 3D representation with quality rivaling NeRF at far higher render speed.

## Functionality

- Per-Gaussian parameters: μ (position), α (opacity), Σ (covariance factorized as rotation R · scale S), SH coefficients for color.
- Scene: 1–5M Gaussians; density control during optimization clones/splits/prunes by gradient magnitude and opacity.

## Used By

| Model | Role |
|-------|------|
| 3D Gaussian Splatting | Scene representation + adaptive density control |
| MonoSplat | Gaussian primitive prediction from fused mono/multi-view features |

## Features

- **Explicit + editable** — unlike NeRF's implicit field.
- **Anisotropic covariance** — flat, oriented Gaussians model surfaces compactly.

## Evolution

- **Predecessor**: NeRF + volume-rendering (implicit, slow); Plenoxels.
- **Successor**: 2DGS, 4DGS (dynamic), scaffold-GS (anchor-structured Gaussians).
