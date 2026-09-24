# YOLOv11

## Overview

YOLOv11 is Ultralytics' one-stage detector/segmenter in the YOLO line, keeping the anchor-free decoupled-head design of YOLOv8 while adding **C3k2** backbone blocks and a **C2PSA** (convolutional block with parallel spatial attention) module in the neck. The headline result is better accuracy **with fewer parameters** — e.g. YOLOv11m uses about 22% fewer parameters than YOLOv8m.

- **Year:** 2024
- **Authors:** Khanam & Hussain (Ultralytics)
- **Source:** arXiv:2410.17725 — *YOLOv11: An Overview of the Key Architectural Enhancements*
- **Category:** DL/Detection

## Key Characteristics

- **C3k2 blocks** replace the older C2f blocks: two convolutions instead of one large one (a "kernel-bottleneck" formulation), which is where most of the parameter saving comes from.
- **C2PSA** — a CSP-style block with **parallel spatial attention** inserted after SPPF, so the neck can re-weight spatial locations before multi-scale fusion.
- **SPPF + PAN neck** retained: fast spatial pyramid pooling, then top-down/bottom-up feature fusion.
- **Anchor-free, decoupled head** with task-aligned assignment; the same backbone/neck serves **detection, segmentation, pose, and OBB** heads.
- Practical focus: deployment-friendly (export to ONNX/TensorRT), strong speed-accuracy trade-off on COCO.
- Note: this is Ultralytics' own overview paper, not a peer-reviewed architecture paper — treat claims (e.g. the 22% parameter reduction) as vendor-reported.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Khanam_et_al._2024_2410.17725.md`](references/papers/Khanam_et_al._2024_2410.17725.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (YOLOv5 → YOLOv8 → YOLOv11 → YOLOv12 / YOLO26; siblings: RT-DETR, D-FINE).
