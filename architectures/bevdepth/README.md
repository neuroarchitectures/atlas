# BEVDepth

## Overview

Camera-only BEV 3D detection depends on depth to lift image features into 3D, and BEVDepth's diagnosis is that this step was **surprisingly inadequate** in prior work — the rest of the pipeline was being optimized while depth estimation stayed weak. BEVDepth fixes depth specifically: explicit supervision, camera awareness, and a refinement module for the errors that imprecise depth introduces downstream.

- **Year:** 2022
- **Authors:** Li et al. (MEGVII Technology, Institute of Computing Technology CAS, UCAS, HUST, Xi'an Jiaotong University)
- **Source:** arXiv:2206.10092 — *BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection*
- **Category:** DL/3D Detection

## Key Characteristics

- **Explicit depth supervision** — depth is supervised directly (from projected LiDAR) rather than being learned only through the detection loss, which is the stated root cause of prior weakness.
- **Camera-aware depth estimation module** — intrinsics/extrinsics are given to the depth network so it can predict depth for the actual camera rather than an implicit one.
- **Depth refinement module** — counteracts the side effects of imprecise feature unprojection, i.e. errors that propagate from depth into the BEV features.
- **Efficient voxel pooling + multi-frame** — a faster frustum-to-BEV pooling operation and temporal fusion.
- **First camera model past 60 NDS** — 60.9% NDS on the nuScenes test set while maintaining high efficiency.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Li_et_al._2022_2206.10092.md`](references/papers/Li_et_al._2022_2206.10092.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (LSS / OFT → BEVDepth; siblings: BEVFormer, PETR, SparseBEV, StreamPETR; successors: BEVDepth4D, SOLOFusion, StreamPETR).
