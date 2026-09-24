# MambaVision

## Overview

First-generation Mamba vision models disappointed: Mamba's causal formulation needs the whole sequence before predicting, which is awkward for images, and in practice **ViT and CNN backbones still outperformed the best Mamba-based vision models**. MambaVision responds by redesigning the Mamba block for vision and then going **hybrid** — several Transformer blocks in the final stages, where they pay for themselves in global context and throughput.

- **Year:** 2024
- **Authors:** Hatamizadeh and Kautz (NVIDIA)
- **Source:** arXiv:2407.08083 — *MambaVision: A Hybrid Mamba-Transformer Vision Backbone*
- **Category:** DL/Vision SSM

## Key Characteristics

- **Redesigned, vision-friendly Mamba block** — the paper's first contribution is a new formulation (MambaVision Mixer) that improves accuracy and image throughput over the original Mamba architecture.
- **Hybrid Mamba + Transformer** — claimed as the first such hybrid architecture for computer vision; integration patterns are studied systematically (earlier / middle / final layers, and every l-th layer).
- **Self-attention at the final stages wins** — the study's finding: putting several self-attention blocks late significantly improves capture of global context and long-range spatial dependencies, and also increases image throughput versus both pure Mamba and ViT models.
- **Multi-resolution architecture** — with **CNN-based residual blocks** for fast extraction of larger-resolution features.
- **New SOTA Pareto front** on ImageNet-1K top-1 versus image throughput, outperforming Mamba-, CNN- and ViT-based models; downstream MS COCO and ADE20 results beat comparably-sized counterparts.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Hatamizadeh_et_al._2024_2407.08083.md`](references/papers/Hatamizadeh_et_al._2024_2407.08083.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Mamba → VMamba → MambaVision; siblings: Vim, LocalMamba; contrast: pure-Mamba vision models and pure ViT backbones).
