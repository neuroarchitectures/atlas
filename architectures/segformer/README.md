# SegFormer

## Overview

Transformer-based segmentation inherited two things from ViT that hurt dense prediction: fixed-resolution positional encoding (which breaks when test resolution differs from training) and a heavy decoder. SegFormer removes both — a **hierarchically structured transformer encoder** that outputs multi-scale features **without positional encoding**, plus an **all-MLP decoder** that aggregates those levels.

- **Year:** 2021
- **Authors:** Xie et al. (NVlabs, University of Illinois, McGill, Caltech, UC Merced)
- **Source:** arXiv:2105.15203 — *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*
- **Category:** DL/Segmentation

## Key Characteristics

- **No positional encoding** — replaced by a **Mix-FFN** (a 3×3 convolution inside the feed-forward network), which sidesteps the interpolation of positional codes that degrades performance at unseen test resolutions.
- **Hierarchical transformer encoder** — outputs multi-scale features rather than a single-resolution map, which dense prediction needs.
- **All-MLP decoder** — no attention in the decoder at all; it upsamples and concatenates multi-level features, combining local and global attention cheaply.
- **Efficiency is the point** — SegFormer-B4 reaches 50.3% mIoU on ADE20K with 64M parameters: 5× smaller and 2.2% better than the prior best.
- **Zero-shot robustness** — SegFormer-B5 reaches 84.0% mIoU on Cityscapes validation and shows strong robustness on Cityscapes-C.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Xie_et_al._2021_2105.15203.md`](references/papers/Xie_et_al._2021_2105.15203.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (FCN / DeepLab / SETR → SegFormer; siblings: Mask2Former, OneFormer, MaskFormer; successors: SegNeXt, InternImage segmentation backbones).
