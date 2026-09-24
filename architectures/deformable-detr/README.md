# Deformable DETR

## Overview

Deformable DETR fixes the two structural weaknesses of DETR: **slow convergence** and **limited feature spatial resolution**. It replaces global dense attention with **deformable (sparse) attention** that samples a small set of points around a learned reference point, and extends the detector to **multi-scale feature maps** — which is what makes small objects work.

- **Year:** 2021
- **Authors:** Zhu et al. (SenseTime Research / USTC)
- **Source:** arXiv:2010.04159 — *Deformable DETR: Deformable Transformers for End-to-End Object Detection* (ICLR 2021)
- **Category:** DL/Detection

## Key Characteristics

- **Deformable attention module**: each query attends to K sampling points (K = 4 in the encoder, K = 4/8 in the decoder, typically 3×3 multi-scale) whose offsets and weights are predicted from the query feature — sparse instead of all-pairs attention.
- **Multi-scale deformable attention** over C3–C5 backbone features; no FPN required, but scales are fused by the attention itself.
- Convergence in a fraction of DETR's schedule (≈50 epochs instead of 500), and large gains on small objects.
- Keeps DETR's set prediction: bipartite Hungarian matching, no anchors, no NMS.
- Optional **iterative bounding-box refinement** and **two-stage** variant (encoder features generate region proposals as queries).
- Cost is linear in the number of pixels, so high-resolution inputs become affordable.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhu_et_al._2021_2010.04159.md`](references/papers/Zhu_et_al._2021_2010.04159.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR → Deformable DETR → DAB-DETR → DINO / RT-DETR).
