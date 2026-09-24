# PVT

## Overview

ViT was designed for image classification specifically, and porting it to dense prediction is difficult: it **yields low-resolution outputs** (bad for pixel-level tasks) and **incurs high computational and memory cost**. PVT is a **convolution-free** backbone built for dense prediction, keeping a CNN-like pyramid so it can replace a CNN backbone directly.

- **Year:** 2021
- **Authors:** Wang et al. (Nanjing University of Science and Technology, The Chinese University of Hong Kong, University of Queensland, IIAI, Nanjing University)
- **Source:** arXiv:2102.12122 — *Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions*
- **Category:** DL/ViT

## Key Characteristics

1. **High output resolution on dense partitions** — unlike ViT, PVT can be trained on dense partitions of an image to achieve high output resolution, which is what dense prediction needs.
2. **Progressive shrinking pyramid** — reduces the computation of large feature maps, addressing the cost problem.
3. **Convolution-free but CNN-compatible** — inherits advantages of both CNN and Transformer, making it a **unified backbone without convolutions** that can be a direct replacement for CNN backbones.
4. **Validated across downstream tasks** — object detection, instance segmentation and semantic segmentation.
5. **Concrete gain** — with a comparable number of parameters, **PVT + RetinaNet achieves 40.4 AP on COCO**, surpassing ResNet50 + RetinaNet (36.3 AP) by **4.1 absolute AP**.

The paper's stated aim is to explore an alternative backbone beyond CNN useful for dense prediction, and to offer PVT as an option for pixel-level prediction research.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2021_2102.12122.md`](references/papers/Wang_et_al._2021_2102.12122.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (CNNs, ViT → PVT; siblings: Swin V2, CSWin, MaxViT, MViTv2; successors: PVT v2 and the pyramid-attention backbone family; contrast: ViT, which is low-resolution and costly for dense prediction).
