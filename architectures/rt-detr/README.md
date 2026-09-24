# RT-DETR

## Overview

RT-DETR is a **real-time, NMS-free** detector: an efficient hybrid encoder handles multi-scale features far more cheaply than a plain transformer encoder, and **uncertainty-minimal query selection** feeds the decoder only high-quality queries. It is the first DETR-style detector to beat comparable YOLOs on both accuracy and latency.

- **Year:** 2024
- **Authors:** Lv et al. (Baidu Inc.)
- **Source:** arXiv:2304.08069 — *DETRs Beat YOLOs on Real-time Object Detection*
- **Category:** DL/Detection

## Key Characteristics

- **Efficient hybrid encoder**: decouples intra-scale interaction (AIFI, self-attention on S5 only) from cross-scale fusion (CCFF, a CNN-based fusion path with 1×1 convolutions and reparameterized blocks).
- **Uncertainty-minimal query selection**: queries are chosen by explicitly minimizing epistemic uncertainty, instead of DETR-style random initialization or a pure classification-score selection.
- **Speed is tunable at inference**: the decoder depth and the number of queries can be adjusted without retraining, giving a direct accuracy/latency dial.
- No anchors and no NMS — post-processing cost is predictable, which matters for real-time deployment.
- Reported on a T4 GPU with TensorRT FP16: RT-DETR-R101 reaches 54%+ AP on COCO at speeds faster than comparable YOLO detectors.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Lv_et_al._2024_2304.08069.md`](references/papers/Lv_et_al._2024_2304.08069.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR / Deformable DETR → RT-DETR; competitive line: YOLO v8–v11).
