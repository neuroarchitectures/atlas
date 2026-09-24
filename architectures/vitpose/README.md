# ViTPose

## Overview

Pose models had accumulated task-specific machinery — hierarchical backbones with multi-scale fusion, elaborate decoders, custom attention — and it was unclear how much of it mattered. ViTPose's answer: use a **plain, non-hierarchical vision transformer** as the backbone and a **simple decoder** (deconvolution layers, or even a single upsampling layer), and let backbone scaling supply the accuracy.

- **Year:** 2022
- **Authors:** Xu et al.
- **Source:** arXiv:2204.12484 — *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*
- **Category:** DL/2D Human Pose

## Key Characteristics

- **Plain non-hierarchical ViT backbone** — no pyramid, no multi-scale fusion stage; the transformer's own capacity is the feature extractor.
- **Simple decoder** — a few deconvolution layers, and the paper shows it can be reduced to a single upsampling layer with modest loss.
- **Heatmap regression** — predicts one heatmap per keypoint, the standard top-down keypoint formulation.
- **Scaling works** — accuracy tracks backbone size, which is the empirical claim: scale the transformer rather than engineering the decoder.
- **Transferable to animals/other keypoints** — the same design applies beyond human pose (ViTPose++ extends the family).

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Xu_et_al._2022_2204.12484.md`](references/papers/Xu_et_al._2022_2204.12484.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (HRNet / SimpleBaseline / HRFormer → ViTPose; siblings: RTMPose, ViTPose++; successors: ViTPose++, transformer-based whole-body models).
