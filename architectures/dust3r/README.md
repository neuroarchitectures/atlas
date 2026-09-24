# DUSt3R

## Overview

DUSt3R (Dense Unconstrained Stereo 3D Reconstruction) turns 3D reconstruction into a **feed-forward regression problem**: given two uncalibrated, unposed images, a siamese ViT encoder plus a cross-attention decoder directly regresses dense **pointmaps** for both views in a shared coordinate frame. Camera poses, intrinsics, and depth fall out of the pointmaps instead of being solved first.

- **Year:** 2023
- **Authors:** Wang et al. (Naver Labs Europe / ENS)
- **Source:** arXiv:2312.14132 — *DUSt3R: Geometric 3D Vision Made Easy*
- **Category:** DL/3D-Reconstruction

## Key Characteristics

- **No camera calibration and no pose prior** required — a strict relaxation of classical stereo/SfM pipelines.
- Siamese **ViT-L encoder** with shared weights, followed by a transformer decoder with **cross-attention** so the two views exchange information.
- Two heads: a **pointmap head** (dense 3D point per pixel, expressed in view-1 coordinates) and a **confidence head** used to weight the regression loss.
- Pointmap regression formulation makes the output usable by many downstream tasks: monocular/multi-view depth, camera pose, intrinsics, dense reconstruction.
- Multi-view extension via **global alignment**: align pairwise pointmaps into a common frame by optimization.
- Limitations: core model is pairwise (cost grows with views), scale can be ambiguous from a single pair, and no explicit camera model is learned.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2023_2312.14132.md`](references/papers/Wang_et_al._2023_2312.14132.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SfM/MVS → DUSt3R → MASt3R, Spann3R, Fast3R, VGGT).
