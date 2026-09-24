# SAM-HQ (HQ-SAM)

## Overview

SAM's masks are geometrically correct but boundary-imprecise: its ViT encoder operates at a fixed stride and the mask decoder has no high-resolution path. SAM-HQ (HQ-SAM) keeps **SAM entirely frozen** and adds a small number of learnable components: a **HQ-Output Token** in the mask decoder, and **HQ-Features** formed by global-local fusion of early high-resolution encoder features with the final features. A short upsampling path turns the result into an accurate mask.

- **Year:** 2023
- **Authors:** Ke et al.
- **Source:** arXiv:2306.01567 — *Segment Anything in High Quality*
- **Category:** DL/Segmentation

## Key Characteristics

- **Frozen SAM** — the pretrained encoder, prompt encoder and mask decoder are untouched; only the new modules are trained.
- **HQ-Output Token** — a learnable token in the mask decoder dedicated to the high-quality mask, so the original coarse output is preserved.
- **Global-local feature fusion** — combines early (high-resolution, local detail) and late (global semantics) ViT features; a simple fusion that beats an FPN-style alternative.
- **HQ mask upsampling** — a small convolutional path that recovers thin structures and fine boundaries.
- **Cheap** — a tiny parameter and latency overhead on top of SAM, unlike a full retrain; zero-shot ability is retained.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Ke_et_al._2023_2306.01567.md`](references/papers/Ke_et_al._2023_2306.01567.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SAM → SAM-HQ; siblings: MobileSAM/EdgeSAM (speed), EfficientSAM (pretraining), SAM 2 (video); downstream: matting, annotation tools).
