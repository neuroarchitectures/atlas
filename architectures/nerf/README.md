# NeRF

## Overview

NeRF addresses the long-standing problem of **view synthesis** in a new way: rather than predicting pixels from features, it **directly optimizes parameters of a continuous 5D scene representation** to minimize the error of rendering a set of captured images. A scene is stored as a fully-connected network — no convolutional layers — whose input is a single continuous 5D coordinate and whose output is volume density and view-dependent radiance.

- **Year:** 2020
- **Authors:** Mildenhall et al. (UC Berkeley, Google Research, UC San Diego)
- **Source:** arXiv:2003.08934 — *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
- **Category:** DL/Neural Scene Representation

## Key Characteristics

- **Continuous 5D scene function** — input is spatial location \((x,y,z)\) plus viewing direction \((\theta,\phi)\); output is **volume density** and **view-dependent emitted radiance**.
- **Fully-connected, non-convolutional** — an MLP represents the function; density acts like a differential opacity controlling how much radiance accumulates along a ray.
- **Differentiable volume rendering** — views are synthesized by marching camera rays, querying the network, and **accumulating colours and densities into a 2D image** with classical volume rendering; because this is naturally differentiable, the only input required is **a set of images with known camera poses**.
- **View-dependence is explicit** — emitted radiance depends on viewing direction, which is what lets complicated appearance be represented.
- **Photorealistic novel views** — results outperform prior work on neural rendering and view synthesis.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Mildenhall_et_al._2020_2003.08934.md`](references/papers/Mildenhall_et_al._2020_2003.08934.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (mesh/voxel-based scene representations, neural rendering → NeRF; successors: 3D Gaussian Splatting family, MonoSplat; contrast: convolutional scene representations, which NeRF deliberately avoids).
