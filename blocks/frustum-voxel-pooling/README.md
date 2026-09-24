# Frustum-to-BEV Voxel Pooling (Lift-Splat)

## Design Philosophy

Convert per-pixel depth distributions into a 3D representation by *lifting*: each pixel's features are replicated along a frustum of depth bins weighted by a predicted depth distribution, then *splatted* into BEV voxels by summation — an explicit, efficient view transform without attention.

## Functionality

- Depth distribution α per pixel (from a depth net) → outer product with image features → frustum point cloud.
- Scatter-add frustum points into BEV/voxel grid (interval-based efficient pooling, not naive O(N·V)); BEVDepth variant uses accurate intrinsics-aware depth.

## Used By

| Model | Role |
|-------|------|
| BEVDepth | Depth-weighted frustum features → BEV grid before multi-frame fusion |

## Features

- **Explicit geometry** — the depth distribution is interpretable and supervisable (lidar-depth-supervision).
- **Hardware-friendly scatter** — the standard 2D→3D workhorse.

## Evolution

- **Predecessor**: Lift-Splat-Shoot (2020).
- **Successor**: attention-based lifting (BEVFormer's bev-query-grid) removes the explicit pooling; depth-refinement modules correct pooling errors.
