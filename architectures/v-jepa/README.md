# V-JEPA

## Overview

V-JEPA applies joint-embedding predictive architecture to video: mask a spatio-temporal region, encode the *visible* context, and **predict the representations of the masked region in feature space**. The prediction target comes from a **target encoder that is an EMA of the context encoder** (with stop-gradient), so there are no pixels to reconstruct, no negative samples, no contrastive loss, and no pretrained image encoder.

- **Year:** 2024
- **Authors:** Bardes et al. (Meta AI)
- **Source:** arXiv:2404.08471 — *Revisiting Feature Prediction for Learning Visual Representations from Video*
- **Category:** DL/Self-Supervised Learning

## Key Characteristics

- **Feature prediction, not pixel prediction** — predicting representations avoids spending capacity on irreducible pixel detail, which the paper argues is the reason feature prediction is more label-efficient.
- **EMA target encoder** — targets are produced by an exponential-moving-average copy of the encoder with stop-gradient; this (not contrastive negatives) prevents collapse.
- **Predictor module** — a narrow network maps context features to predicted target features; it is discarded at evaluation, only the encoder is used.
- **Spatio-temporal masking** — tubes are masked so the task requires temporal reasoning, not just spatial inpainting.
- **Frozen evaluation** — representations are assessed by training lightweight probes on top of a frozen encoder.
- **No pretrained image encoder** — unlike VideoMAE-style or CLIP-initialized pipelines, the model is trained from scratch on video.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Bardes_et_al._2024_2404.08471.md`](references/papers/Bardes_et_al._2024_2404.08471.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (I-JEPA → V-JEPA → V-JEPA 2 / 2.1; siblings: VideoMAE (pixel reconstruction), MAE, DINO (contrastive/self-distillation), DINOv2).
