# BEVFormer

## Overview

BEVFormer turns **multi-camera images into a bird's-eye-view (BEV) representation** that directly supports 3D detection and map segmentation, without depth supervision. A grid of **BEV queries** gathers information by **spatial cross-attention** over the multi-view features and by **temporal self-attention** over the previous BEV map — the two mechanisms that replace explicit depth estimation.

- **Year:** 2022
- **Authors:** Li et al. (Nanjing University / Shanghai AI Laboratory / SenseTime)
- **Source:** arXiv:2203.17270 — *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*
- **Category:** DL/3D-Detection

## Key Characteristics

- **Grid-shaped BEV queries** (e.g. 200×200) — each query cell owns a region of the ground plane, so the output is a structured BEV feature map rather than a set of object slots.
- **Spatial cross-attention with deformable sampling** — each BEV query lifts to a column of heights and samples the corresponding locations in the camera feature maps, so no depth network or LiDAR is needed.
- **Temporal self-attention** — the current BEV queries attend to the previous frame's BEV features (recurrent), giving velocity estimation, better occlusion reasoning, and temporal smoothing.
- **Multi-task heads** on the same BEV map: 3D boxes, velocities, and map segmentation.
- Reported to improve NDS on the nuScenes test set by about **9 points** over the prior art.
- Cost: two attention passes over a large BEV grid plus a per-view backbone — the BEV grid resolution is the main latency/accuracy dial.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Li_et_al._2022_2203.17270.md`](references/papers/Li_et_al._2022_2203.17270.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (LSS / BEVDet / DETR3D → BEVFormer → BEVFormer v2, StreamPETR, SparseBEV, OccTransformer).
