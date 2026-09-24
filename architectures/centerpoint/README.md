# CenterPoint

## Overview

3D detection borrowed the 2D anchor-box formulation, but a 3D box does not fit a rotated object well and anchors must then enumerate orientations — extra compute and extra false positives. CenterPoint drops boxes for the detection representation: objects are **points**. A keypoint detector finds centers and regresses the rest; tracking then becomes greedy closest-point matching.

- **Year:** 2020
- **Authors:** Yin et al. (UT Austin, NVIDIA)
- **Source:** arXiv:2006.11275 — *Center-based 3D Object Detection and Tracking*
- **Category:** DL/3D Detection

## Key Characteristics

- **Objects as points** — the first stage detects object centers with a keypoint detector and regresses 3D size, orientation and velocity, which removes the orientation-enumeration problem entirely.
- **Second-stage refinement** — point features at the object refine the estimates; the paper shows bird's-eye-view features are sufficient for good performance and more efficient than voxel features.
- **Tracking is nearest-neighbor** — with a learned point velocity, tracking simplifies to greedy closest-point matching with no hidden state: 1 ms versus 73 ms for a 3D Kalman filter, and better (3.7 AMOTA) than Mahalanobis matching.
- **Simple and near real-time** — a standard 3D point-cloud encoder plus a few convolutional head layers.
- **Results** — 65.5 NDS and 63.8 AMOTA on nuScenes; first among Lidar-only submissions on Waymo, with +18.6 mAPH on the hard pedestrian class.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Yin_et_al._2020_2006.11275.md`](references/papers/Yin_et_al._2020_2006.11275.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (VoxelNet / PointPillars / PV-RCNN anchor-based → CenterPoint; siblings: PillarNet, VoxelNeXt, CenterPoint++; contrast: camera-BEV methods PETR, BEVDepth, BEVFormer).
