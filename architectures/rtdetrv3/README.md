# RT-DETRv3

## Overview

RT-DETRv3 targets the remaining weakness of real-time DETRs: one-to-one matching gives each decoder layer **too few positive samples** to learn from. It adds **hierarchical dense positive supervision** — auxiliary branches with denser (one-to-many) assignment attached at several depths — so the real-time model gets supervision as rich as a slower detector while keeping the same inference graph.

- **Year:** 2024
- **Authors:** Wang et al. (Baidu VIS / Baidu Inc.)
- **Source:** arXiv:2409.08475 — *RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision*
- **Category:** DL/Detection

## Key Characteristics

- **Hierarchical dense positive supervision** — auxiliary supervision branches are attached at multiple levels (encoder/decoder depths), each using a denser assignment than the main one-to-one matching.
- **Training-only branches**: the auxiliary modules are discarded at inference, so the deployment graph is unchanged — this is the design constraint that makes it a real-time method.
- Built on **RT-DETR**: CNN backbone + hybrid encoder (AIFI intra-scale attention + CCFM cross-scale fusion) + query selection + decoder.
- Reported to meaningfully improve AP over RT-DETR at the same FPS on COCO.
- Shares the general lesson with Group-DETR / DEYO / DINO: one-to-many assignment is a *training* trick that can be removed at test time.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2024_2409.08475.md`](references/papers/Wang_et_al._2024_2409.08475.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR → Deformable DETR → DINO → RT-DETR → RT-DETRv2 → RT-DETRv3; siblings: D-FINE, LW-DETR, YOLOv11).
