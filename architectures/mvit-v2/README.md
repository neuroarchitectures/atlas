# MViTv2

## Overview

Visual signals are dense, so self-attention's **quadratic** compute and memory scaling is severe for high-resolution detection and space-time video. Two strategies had emerged: **local window attention** for detection, and **pooling attention** (locally aggregating before self-attention) for video — the latter fuelling Multiscale Vision Transformers, which had a feature hierarchy from high to low resolution rather than ViT's fixed resolution. MViTv2 improves pooling attention and then asks whether one model family can serve **image classification, object detection and video classification** as a general backbone.

- **Year:** 2021
- **Authors:** Li et al. (Facebook AI Research (FAIR), UC Berkeley)
- **Source:** arXiv:2112.01526 — *MViTv2: Improved Multiscale Vision Transformers for Classification and Detection*
- **Category:** DL/ViT

## Key Characteristics

1. **Decomposed relative position embeddings** — shift-invariant positional embeddings using **decomposed location distances** to inject position information into Transformer blocks.
2. **Residual pooling connection** — compensates for the effect of pooling strides in attention computation.
3. **Two simple improvements, large effect** — the paper emphasizes these upgrades are simple yet significantly better.
4. **Standard dense prediction framework** — Mask R-CNN with FPN applied to object detection and instance segmentation on top of the improved structure.
5. **One family across spatial and spatiotemporal tasks** — studied explicitly as a general vision backbone for both, which is the paper's framing question.
6. **Kinetics result** — **86.1%** on Kinetics-400 video classification.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Li_et_al._2021_2112.01526.md`](references/papers/Li_et_al._2021_2112.01526.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (ViT, MViT → MViTv2; siblings: Swin V2, PVT, CSWin, MaxViT, VideoMAE; contrast: local-window-attention backbones, the other strategy for the quadratic-cost problem).
