# CSWin Transformer

## Overview

Transformer backbone design hits a dilemma: **global self-attention is very expensive**, while **local self-attention limits each token's field of interaction**. Window-based designs bridge windows with halo and shift operations, but the receptive field then grows quite slowly. CSWin addresses this with **Cross-Shaped Window** self-attention — horizontal and vertical stripes computed in parallel forming a cross — with the stripe width analyzed mathematically and varied per layer.

- **Year:** 2021
- **Authors:** Dong et al. (USTC, MSRA, Microsoft Cloud + AI)
- **Source:** arXiv:2107.00652 — *CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows*
- **Category:** DL/ViT

## Key Characteristics

- **Cross-Shaped Window self-attention** — self-attention computed in **horizontal and vertical stripes in parallel**, each stripe obtained by splitting the input feature into stripes of equal width, together forming a cross-shaped window.
- **Mathematical analysis of stripe width** — the effect of stripe width is analyzed, and the width is varied across layers to get strong modeling capability while limiting computation cost.
- **Locally-enhanced Positional Encoding (LePE)** — handles local positional information better than existing encoding schemes; naturally supports **arbitrary input resolutions**, making it effective and friendly for downstream tasks.
- **Hierarchical structure** — incorporated with the above into a general backbone.
- **Surpasses Swin at similar FLOPs** — **85.4% ImageNet-1K top-1** with no extra data or label; 53.9 box AP and 46.4 mask AP on COCO; **52.2 mIoU** on ADE20K — surpassing the previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4 and +2.0 respectively at similar FLOPs.
- **Scales with data** — ImageNet-21K pre-training gives **87.5% top-1** on ImageNet-1K and **55.7 mIoU** on ADE20K.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Dong_et_al._2021_2107.00652.md`](references/papers/Dong_et_al._2021_2107.00652.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (ViT, Swin Transformer → CSWin; siblings: Swin V2, PVT, MaxViT, MViTv2; contrast: halo/shift window-bridging designs, whose receptive field grows slowly).
