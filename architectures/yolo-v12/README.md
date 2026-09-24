# YOLOv12

## Overview

YOLOv12 is the first YOLO release built around **attention** rather than CNNs, while staying real-time. Three pieces make it work: **R-ELAN** (residual efficient layer aggregation, fixing the optimization instability of plain attention blocks), **area attention** (large receptive field at reduced attention cost), and **position-aware separable convolutions** (7×7-style depthwise separable convs supplying the positional signal that attention lacks).

- **Year:** 2025
- **Authors:** Tian et al.
- **Source:** arXiv:2502.12524 — *YOLOv12: Attention-Centric Real-Time Object Detectors*
- **Category:** DL/Detection

## Key Characteristics

- **R-ELAN** — ELAN blocks with a block-level residual plus a bottleneck, which is what makes a deep attention-centric stack converge.
- **Area attention** — splits the feature map into areas (horizontal/vertical) instead of attending globally or with fixed windows: large receptive field, lower complexity than vanilla self-attention.
- **Position-aware separable convolution** — depthwise separable conv with a large kernel, replacing positional encodings with a cheaper inductive bias.
- **Real-time despite attention** — FlashAttention-compatible implementation and reduced-complexity attention keep latency competitive with CNN YOLOs.
- **Same head family** — decoupled anchor-free head and multi-task support (detect / segment / pose / OBB) continue from YOLOv11.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Tian_et_al._2025_2502.12524.md`](references/papers/Tian_et_al._2025_2502.12524.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (YOLOv8 → YOLOv11 → YOLOv12 → YOLO26; siblings: RT-DETR, D-FINE, LW-DETR; inspiration: ELAN/GELAN from YOLOv9).
