# PETR

## Overview

DETR3D gave multi-view 3D detection an end-to-end DETR formulation, but each decoder layer had to project the query's predicted 3D reference point back into image space and sample features there — inaccurate reference points sample outside the object, sampling only one point misses global context, and the sampling procedure itself is awkward in practice. PETR removes the projection: bake **3D position information into the 2D features** so queries can be updated directly in 3D.

- **Year:** 2022
- **Authors:** Liu et al. (MEGVII Technology, Nanjing University, Beihang, Tsinghua, Shanghai AI Laboratory)
- **Source:** arXiv:2203.05625 — *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*
- **Category:** DL/3D Detection

## Key Characteristics

- **Position embedding transformation** — 3D coordinates of the camera frustum space are encoded and added to the 2D image features, producing **3D position-aware features**; no online 2D-to-3D transformation and no feature sampling.
- **Inspired by implicit neural representations** — the same trick MetaSR and LIFF use to encode high-resolution coordinates into low-resolution features.
- **End-to-end DETR decoder** — object queries perceive the 3D-aware features and are updated directly in the 3D environment.
- **Simple and strong baseline** — 50.4% NDS and 44.1% mAP on nuScenes, ranked 1st on the benchmark at publication; explicitly positioned as a baseline for future work.
- **Removes a failure mode** — because there is no projected reference point, there is no dependence on the accuracy of a predicted coordinate.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2022_2203.05625.md`](references/papers/Liu_et_al._2022_2203.05625.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (monocular 3D → DETR3D → PETR; siblings: BEVDepth, BEVFormer, SparseBEV, StreamPETR; successors: PETRv2 (temporal), StreamPETR).
