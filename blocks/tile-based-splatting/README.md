# Tile-Based Splatting Rasterizer

## Design Philosophy

Render Gaussians like a graphics pipeline, not a ray marcher: project all Gaussians to screen, sort them by depth per 16×16 tile, then alpha-blend front-to-back with anisotropic 2D covariance footprints. Sorted traversal also yields the exact backward pass — differentiable rasterization at real-time rates.

## Functionality

- Frustum culling → camera projection of anisotropic covariance → global depth sort → per-tile parallel α-blending (`C = Σ c_i α_i Π(1-α_j)`).
- Backward: reverse sorted traversal recomputes blending weights exactly.

## Used By

| Model | Role |
|-------|------|
| 3D Gaussian Splatting | 1080p rendering ≥30 fps; exact gradients via sorted traversal |
| MonoSplat | Rendering fused-feature-predicted Gaussians |

## Features

- **Real-time differentiable rendering** — the performance unlock of 3DGS.
- **Exact α-blending order** — correct occlusion without ray marching.

## Evolution

- **Predecessor**: classical surface splatting (Zwicker 2001); NeRF volume rendering.
- **Successor**: 2DGS surfels, Mip-Splatting (anti-aliasing), and hardware rasterizer ports.
