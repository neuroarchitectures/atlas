# DINO (DETR with Improved DeNoising Anchor Boxes)

## Overview

DINO is a DETR-family detector that makes end-to-end detection competitive with — and better than — the best classical detectors. Three ideas do the work: **contrastive denoising training** (feed both positive and negative noised boxes and learn to reject the negatives), **mixed query selection**, and **look forward twice** box refinement. It is the accuracy reference point for the DETR line and the basis of Grounding DINO.

- **Year:** 2022
- **Authors:** Zhang et al. (Tsinghua University / IDEA Research)
- **Source:** arXiv:2203.03605 — *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection*
- **Category:** DL/Detection

## Key Characteristics

- **Contrastive denoising (CDN)**: besides the usual reconstruction of noised ground-truth boxes, explicitly add **negative** noised boxes and supervise the model to predict "no object" for them — this is what removes duplicate predictions.
- **Mixed query selection**: anchor queries come from the encoder (positional queries) while content queries stay learnable, giving better initialization than either pure-static or pure-dynamic selection.
- **Look forward twice**: a later layer's box prediction supervises the refinement of earlier layers, correcting the greedy per-layer updates.
- Deformable multi-scale attention backbone (Swin-L or ResNet); still **NMS-free** and anchor-free.
- Reported **63.3 AP on COCO test-dev**; trains far faster than the original DETR and is the first DETR model to top the COCO leaderboard.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhang_et_al._2022_2203.03605.md`](references/papers/Zhang_et_al._2022_2203.03605.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR → Deformable DETR → DAB-DETR / DN-DETR → DINO → Grounding DINO, RT-DETR, D-FINE).
