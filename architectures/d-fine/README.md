# D-FINE

## Overview

D-FINE attacks the part of DETR that is usually left alone: **box regression**. Rather than predicting a single box delta per layer, it treats each edge as a probability distribution and **refines it layer by layer** (Fine-grained Distribution Refinement, FDR), plus a self-distillation term (GO-LSD) that transfers the final layer's localization quality to earlier layers. Accuracy improves with no added inference cost.

- **Year:** 2024
- **Authors:** Peng et al. (China University of Mining and Technology / Tencent)
- **Source:** arXiv:2410.13842 — *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*
- **Category:** DL/Detection

## Key Characteristics

- **Fine-grained Distribution Refinement (FDR)** — each decoder layer predicts a probability distribution over discretized offsets for each box edge and iteratively refines the previous layer's distribution, so localization improves progressively through the decoder.
- **Global Optimal Localization Self-Distillation (GO-LSD)** — the final layer's refined distributions teach earlier layers, which is what makes intermediate layers good enough to matter and speeds convergence.
- Non-monotonic weighting of the distribution makes small adjustments cheap and fine; the distribution form also exposes uncertainty about each edge.
- Built on a standard multi-scale DETR encoder/decoder; **no extra parameters or inference latency** compared with the base detector.
- Reports state-of-the-art speed-accuracy trade-off among real-time DETR variants on COCO.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Peng_et_al._2024_2410.13842.md`](references/papers/Peng_et_al._2024_2410.13842.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR → Deformable DETR → DINO → D-FINE; siblings: RT-DETR / RT-DETRv3, LW-DETR, DEYO; contrast: YOLOv11).
