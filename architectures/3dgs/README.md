# 3D Gaussian Splatting

## Overview

Radiance-field methods had produced excellent novel-view synthesis, but the trade looked unavoidable: high visual quality required costly MLP-based ray marching (Mip-NeRF360 needs up to 48 hours of training), while the faster methods traded speed for quality. **No method reached real-time display rates for unbounded complete scenes at 1080p.** 3DGS breaks that trade with three components: 3D Gaussians as the representation, interleaved optimization with adaptive density control, and a visibility-aware tile-based splatting renderer.

- **Year:** 2023
- **Authors:** Kerbl et al. (Inria, Université Côte d'Azur; Max-Planck-Institut für Informatik / Saarbrücken Research Center for Visual Computing, Robotics and AI)
- **Source:** arXiv:2308.04079 — *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
- **Category:** DL/Neural Scene Representation

## Key Characteristics

- **3D Gaussians as the scene representation** — initialized from the **sparse SfM point cloud produced for free during camera calibration**; differentiable and volumetric like continuous representations, but rasterizable by projection and α-blending, so no sampling of empty space.
- **Interleaved optimization and adaptive density control** — 3D position, opacity α, **anisotropic covariance**, and spherical-harmonic coefficients are optimized while Gaussians are added and occasionally removed, yielding a compact unstructured representation (**1–5 million Gaussians** for all tested scenes).
- **Fast visibility-aware rendering** — tile-based rasterization with fast GPU sorting, supporting **anisotropic splatting** that respects visibility ordering, plus a fast backward pass tracking sorted splat traversal.
- **Real-time at 1080p** — state-of-the-art visual quality with competitive training times and **≥30 fps 1080p** novel-view synthesis.
- **Only SfM points needed** — unlike most point-based solutions, it does not require Multi-View Stereo data.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Kerbl_et_al._2023_2308.04079.md`](references/papers/Kerbl_et_al._2023_2308.04079.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (NeRF and voxel/hash/point radiance fields → 3DGS; siblings: Mono-splat, NeRF variants; contrast: MLP ray-marching methods such as Mip-NeRF360).
