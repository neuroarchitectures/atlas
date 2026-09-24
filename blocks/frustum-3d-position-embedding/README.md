# 3D Frustum Position Embedding

## Design Philosophy

2D image features don't know where they are in 3D space — but in a calibrated multi-camera rig, every pixel *does* have 3D coordinates (a camera frustum). Encode those frustum coordinates into position embeddings and add them to the 2D features up front, making them 3D-aware with no decoder-side projection or sampling.

## Functionality

- Per-view: unproject each pixel via intrinsics/extrinsics to 3D frustum coordinates → MLP → position embedding, added to backbone + FPN features.
- The DETR decoder then operates on 3D-aware features with plain attention.

## Used By

| Model | Role |
|-------|------|
| PETR | Multi-camera 3D object detection; v2 adds per-level embeddings |

## Features

- **One-time encoding** — 3D awareness at zero decoder cost (vs. deformable projection per query).
- **Composable** — plain DETR decoder unchanged on top.

## Evolution

- **Predecessor**: DETR3D (feature sampling along rays per query).
- **Successor**: bev-query-grid and sparse sampling families make the geometry interactive rather than baked-in.
