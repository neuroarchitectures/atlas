# BEV Query Grid

## Design Philosophy

Lift 2D camera features to 3D by asking the question in 3D space: place a learnable query on every cell of an ego-centric ground-plane grid, and let each query *fetch* evidence from the cameras. The grid, not the image, is the persistent representation — and it doubles as recurrent state across frames.

## Functionality

- 200×200 learnable queries + learnable BEV positional embedding.
- Spatial cross-attention: each BEV query projects reference points (a column of heights) into every camera, attends where visible; temporal self-attention fetches from the previous BEV map.
- Shared grid feeds detection and map-segmentation heads.

## Used By

| Model | Role |
|-------|------|
| BEVFormer | Multi-camera 3D detection + map segmentation; recurrent BEV state |

## Features

- **Query-in-world-space** — geometric reasoning without explicit depth estimation.
- **Grid as recurrent state** — see temporal-bev-attention.

## Evolution

- **Predecessor**: Lift-Splat-Shoot (BEVDepth's frustum-voxel-pooling) — explicit pooling into BEV.
- **Successor**: sparsebev (query-adaptive sparse sampling, no dense grid); query-memory-queue (StreamPETR) drops the grid entirely.
