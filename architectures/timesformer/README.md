# TimeSformer

## Overview

Video understanding shares key similarities with NLP — both are sequential, and atomic actions need contextualizing with the rest of the video — yet **2D or 3D convolutions remained the core operators** for spatiotemporal feature learning. Self-attention had only been applied *on top of* convolutional layers. TimeSformer is **convolution-free**, built **exclusively on self-attention over space and time**, learning directly from a sequence of frame-level patches.

- **Year:** 2021
- **Authors:** Bertasius et al. (Facebook AI, Dartmouth College, Indiana University)
- **Source:** arXiv:2102.05095 — *Is Space-Time Attention All You Need for Video Understanding?*
- **Category:** DL/Video Recognition

## Key Characteristics

- **Convolution-free** — built exclusively on self-attention over space and time; not attention on top of conv features.
- **Frame-level patches** — spatiotemporal feature learning directly from a sequence of frame-level patches, following the ViT tokenisation strategy.
- **Divided attention wins** — an experimental study compares different self-attention schemes and finds that **divided attention**, where **temporal and spatial attention are separately applied within each block**, gives the best video classification accuracy among the designs considered.
- **Efficiency stems from the decomposition** — the approach's efficiency comes mainly from decomposing video into frame-level patches rather than from sparse or axial computation tricks.
- **State-of-the-art on action recognition** — including the best reported accuracy on Kinetics-400 and Kinetics-600.
- **Practical advantages over 3D CNNs** — faster to train, dramatically higher test efficiency at a small accuracy drop, and applicable to **much longer video clips (over one minute)**.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Bertasius_et_al._2021_2102.05095.md`](references/papers/Bertasius_et_al._2021_2102.05095.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (two-stream networks, 3D CNNs, ViT → TimeSformer; siblings: ViViT, SlowFast, UniFormer, VideoMAE; contrast: 3D convolutional networks, which remain the comparison point for training speed, test efficiency and clip length).
