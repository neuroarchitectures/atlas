# ViViT

## Overview

Following ViT, attention-based architectures are an intuitive choice for modelling long-range contextual relationships in video. But pure-transformer models present **different characteristics** from the convolutional models the community has years of best practices for, so the design choices must be determined afresh. ViViT develops several transformer-based models for video classification and determines those choices by thorough ablation.

- **Year:** 2021
- **Authors:** Arnab et al. (Google Research, University of Oxford, VGG)
- **Source:** arXiv:2103.15691 — *ViViT: A Video Vision Transformer*
- **Category:** DL/Video Recognition

## Key Characteristics

- **Factorised spatial and temporal dimensions** — ViViT leverages factorisation of the spatial and temporal dimensions of video to increase efficiency, but **in the context of transformer-based models** rather than convolutions.
- **Thorough ablation of the design space** — tokenisation strategies, model architecture and regularisation methods are ablated, since best practices from convolutional models do not transfer.
- **Informed by the ablation** — the resulting configuration achieves state-of-the-art results on multiple standard video classification benchmarks.
- **Trainable on comparatively small datasets** — although transformer-based models are known to be effective only with large training datasets, ViViT shows how to **regularise the model during training** and **leverage pretrained image models** to train on smaller datasets.
- **Breadth of validation** — Kinetics 400 and 600, Epic Kitchens 100, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Arnab_et_al._2021_2103.15691.md`](references/papers/Arnab_et_al._2021_2103.15691.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (two-stream networks, 3D CNNs, ViT → ViViT; siblings: TimeSformer, SlowFast, UniFormer, VideoMAE; contrast: deep 3D convolutional architectures, which were the prior state of the art and remain the baseline outperformed).
