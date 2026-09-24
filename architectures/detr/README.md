# DETR

## Overview

DEtection TRansformer — object detection as a **direct set prediction** problem. A CNN backbone plus a transformer encoder-decoder emits a fixed-size set of boxes and labels in one pass, removing anchors, region proposals, and non-maximum suppression from the detection pipeline.

- **Year:** 2020
- **Authors:** Carion et al. (Facebook AI Research)
- **Source:** arXiv:2005.12872 — *End-to-End Object Detection with Transformers* (ECCV 2020)
- **Category:** DL/Detection

## Key Characteristics

- Set-based global loss with **bipartite (Hungarian) matching** forces one-to-one prediction/ground-truth assignment, so no duplicate suppression is needed.
- **Object queries** (N = 100 learned positional embeddings) act as learned "slots"; the decoder decodes them into detections in parallel.
- Global self-attention in the encoder lets the model reason about all objects and image context jointly.
- Accuracy and run-time on par with a well-optimized Faster R-CNN baseline on COCO, with a far simpler pipeline.
- Costs: long training schedule (500 epochs), weak small-object performance, quadratic attention over the flattened feature map.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Carion_et_al._2020_2005.12872.md`](references/papers/Carion_et_al._2020_2005.12872.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Faster R-CNN → DETR → Deformable DETR → DAB-DETR/DINO → RT-DETR / Grounding DINO).
