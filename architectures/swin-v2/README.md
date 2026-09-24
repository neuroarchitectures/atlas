# Swin Transformer V2

## Overview

NLP models had scaled past 530B dense parameters with few-shot capabilities, while vision models had only just reached 1–2B parameters — and, unlike their NLP counterparts, **existing large vision models were applied to image classification only**. Swin V2 aims at large-scale vision models and names three blocking issues: **training instability**, **resolution gaps between pre-training and fine-tuning**, and **hunger on labelled data**.

- **Year:** 2021
- **Authors:** Liu et al. (MSRA, University of Science and Technology of China, University of North Carolina at Chapel Hill)
- **Source:** arXiv:2111.09883 — *Swin Transformer V2: Scaling Up Capacity and Resolution*
- **Category:** DL/ViT

## Key Characteristics

1. **Residual-post-norm combined with cosine attention** — addresses training instability.
2. **Log-spaced continuous position bias** — enables effective transfer of models pre-trained at low resolution to downstream tasks with high-resolution inputs.
3. **SimMIM self-supervised pre-training** — reduces the need for vast labelled images.
4. **Scale achieved** — a **3 billion-parameter** Swin Transformer V2, the largest dense vision model at the time, capable of training with images up to **1,536×1,536** resolution.
5. **New records on four representative vision tasks** — ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification.
6. **Efficient training** — consumes **40× less labelled data and 40× less training time** than Google's billion-level visual models.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2021_2111.09883.md`](references/papers/Liu_et_al._2021_2111.09883.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Swin Transformer → Swin V2; siblings: PVT, CSWin, MaxViT, MViTv2; contrast: classification-only large vision models that Swin V2 is distinguished from).
