# SparseBEV

## Overview

SparseBEV is a fully sparse multi-camera 3D detector: queries sample a small set of points in 3D space and aggregate image features there, with **scale-adaptive self attention** (sample across scales and adapt sampling offsets per query) and **temporal self attention** (fuse short-term and long-term history). Dense depth estimation is used only to guide *where* to sample, not to build a dense BEV grid.

- **Year:** 2023
- **Authors:** Liu et al.
- **Source:** arXiv:2308.09244 — *SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos*
- **Category:** DL/3D Detection

## Key Characteristics

- **Fully sparse pipeline** — no dense BEV feature map is constructed at any stage; the cost of a frame is the cost of the queries.
- **Scale-adaptive self attention** — each query learns its own sampling pattern over multiple feature scales, replacing exhaustive multi-scale search.
- **Temporal self attention** — fuses a short-term (adjacent frames) and long-term (distant history) stream, giving both motion and context.
- **Depth-distribution guidance** — a depth branch predicts where informative features are, so sampling can concentrate on surfaces instead of empty space.
- **Speed-accuracy trade-off** — the sparse formulation targets high FPS at competitive nuScenes accuracy versus dense BEV baselines.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2023_2308.09244.md`](references/papers/Liu_et_al._2023_2308.09244.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (LSS / BEVDepth / BEVFormer → SparseBEV; siblings: StreamPETR, Sparse4D, DETR3D; contrast: dense-view-transform methods).
