# VGGT

## Overview

VGGT (Visual Geometry Grounded Transformer) is a **feed-forward 3D foundation model**: given one to hundreds of views, a single forward pass predicts **all** key 3D attributes — camera parameters (intrinsics and extrinsics), depth maps, dense point maps, and 3D point tracks. It uses minimal 3D inductive bias: a standard large transformer with **alternating attention** and a multi-task loss.

- **Year:** 2025
- **Authors:** Wang et al. (Visual Geometry Group, University of Oxford / Meta AI)
- **Source:** arXiv:2503.11651 — *VGGT: Visual Geometry Grounded Transformer*
- **Category:** DL/3D-Reconstruction

## Key Characteristics

- **One forward pass, no optimization**: no bundle adjustment or global alignment needed for many tasks, unlike DUSt3R/MASt3R pipelines.
- **Alternating attention**: alternates frame-local (within-image) self-attention and global (across all views) self-attention, giving a token-efficient way to do multi-view reasoning.
- **Minimal 3D priors** — no epipolar geometry module, no cost volume; geometry is learned from data.
- **Multi-task output heads**: camera, depth, point map, and track heads, all supervised jointly.
- Handles arbitrary numbers of views including the single-view case (monocular depth and camera estimation).
- Constraint: attention over all views is quadratic in view count × tokens; long sequences are the memory bottleneck.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2025_2503.11651.md`](references/papers/Wang_et_al._2025_2503.11651.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DUSt3R → MASt3R → VGGT; successors: VGGT-World as an autoregressive geometry world model; siblings: Fast3R, π³).
