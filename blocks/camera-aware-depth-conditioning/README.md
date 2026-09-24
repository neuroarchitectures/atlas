# Camera-Aware Depth Conditioning

## Design Philosophy

Monocular depth networks that ignore camera geometry learn a "typical camera" prior and mispredict on any other. Make depth prediction *conditioned on the actual camera*: feed intrinsics/extrinsics explicitly into the depth network (BEVDepth), or predict a camera tensor that prompts the features (UniDepth) — the network then predicts depth for *this* camera.

## Functionality

- BEVDepth: depth net input = image features + camera intrinsics/extrinsics embeddings; auxiliary depth supervised by projected LiDAR points.
- UniDepth: per-pixel azimuth/elevation camera tensor (Laplace spherical-harmonic embedded) conditions depth features; output in pseudo-spherical form → metric depth.

## Used By

| Model | Role |
|-------|------|
| BEVDepth | Intrinsics/extrinsics-conditioned depth net before frustum-voxel-pooling |
| UniDepth | Self-prompting camera tensor drives metric depth without calibration |

## Features

- **Geometry enters the network** — the fix for cross-camera generalization.
- **Auxiliary depth supervision** (BEVDepth) anchors the depth distribution.

## Evolution

- **Predecessor**: depth-agnostic monocular depth nets; canonical-camera-transform (metric3d) as the alternative (normalize instead of condition).
- **Related**: metric3d — canonicalization makes the network camera-blind instead of camera-aware.
