# RAFT

## Overview

Recurrent All-Pairs Field Transforms — optical flow as an **iterative refinement** problem. Per-pixel features from both frames feed an all-pairs 4D correlation volume; a small weight-tied recurrent update operator looks up the volume and repeatedly updates a single high-resolution flow field, which is finally lifted to full resolution by convex upsampling.

- **Year:** 2020
- **Authors:** Teed & Deng (Princeton University)
- **Source:** arXiv:2003.12039 — *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow* (ECCV 2020, Best Paper)
- **Category:** DL/Motion

## Key Characteristics

- **All-pairs correlation volume** (H×W×H×W) plus a **correlation pyramid** (4 levels) gives both fine matching detail and large-displacement context; no warping-based cost volume.
- **Weight-tied ConvGRU update block** (≈2.7M parameters) can be applied 100+ times at inference without divergence — depth is decoupled from parameter count.
- Single **fixed high-resolution** flow field (1/8 input resolution) instead of a coarse-to-fine cascade: avoids missing small fast-moving objects and avoids multi-stage training.
- **Convex upsampling**: learned convex combination of the 3×3 neighborhood lifts 1/8-resolution flow to full resolution.
- Strong cross-dataset generalization and high efficiency: 1088×436 at 10 fps on a 1080Ti; a version with 1/5 of the parameters runs at 20 fps and still beats prior work on Sintel.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Teed_Deng_2020_2003.12039.md`](references/papers/Teed_Deng_2020_2003.12039.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (FlowNet → SpyNet/PWC-Net → RAFT → FlowFormer, GMFlow, SEA-RAFT).
