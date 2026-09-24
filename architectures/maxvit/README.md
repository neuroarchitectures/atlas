# MaxViT

## Overview

Self-attention does not scale with image size, which had limited its adoption in state-of-the-art vision backbones. MaxViT's answer is **multi-axis attention**: blocked local attention plus dilated global attention, which together allow global-local spatial interactions **at arbitrary input resolutions with only linear complexity**. Blended with convolutions and repeated over multiple stages, the result sees globally throughout the entire network — even in early, high-resolution stages.

- **Year:** 2022
- **Authors:** Tu et al. (Google Research / Brain, University of Michigan, Johns Hopkins University, University of Toronto)
- **Source:** arXiv:2204.01697 — *MaxViT: Multi-Axis Vision Transformer*
- **Category:** DL/ViT

## Key Characteristics

- **Multi-axis attention** — two aspects: **blocked local** and **dilated global** attention.
- **Linear complexity, arbitrary resolution** — global-local spatial interactions are obtained at any input resolution without quadratic cost.
- **Global from the start** — MaxViT "sees" globally throughout the entire network, including earlier high-resolution stages, which is where window-based backbones do not.
- **Blends attention with convolution** — a new architectural element, not pure attention.
- **Simple and hierarchical** — the backbone is a basic building block repeated over multiple stages.
- **Strong results across settings** — **86.5% ImageNet-1K top-1** without extra data; **88.7%** with ImageNet-21K pre-training; favorable object detection and visual aesthetic assessment; strong generative modeling capability on ImageNet, demonstrating the block's potential as a universal vision module.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Tu_et_al._2022_2204.01697.md`](references/papers/Tu_et_al._2022_2204.01697.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (ConvNets, ViT, window-attention backbones → MaxViT; siblings: Swin V2, PVT, CSWin, MViTv2; contrast: local-only attention backbones that do not see globally in early high-resolution stages).
