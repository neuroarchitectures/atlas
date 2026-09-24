# ViT-Adapter

## Overview

Plain ViT has a decisive advantage over vision-specific transformers: **no assumption about the input data**, so with different tokenizers it can be pre-trained on massive multi-modal data — image, video and text — learning semantic-rich representations. But it also has conclusive defects in dense prediction: lacking image-related prior knowledge it converges slower and performs lower, so plain ViT cannot compete with vision-specific transformers on detection and segmentation. ViT-Adapter closes that gap **without modifying the ViT and without re-pre-training**.

- **Year:** 2022
- **Authors:** Chen et al. (Tsinghua University, SenseTime Research, Shanghai AI Laboratory, University of Macau)
- **Source:** arXiv:2205.08534 — *Vision Transformer Adapter for Dense Predictions*
- **Category:** DL/ViT Adaptation

## Key Characteristics

- **Pre-training-free additional network** — adapts a plain ViT to downstream dense prediction tasks **without modifying its original architecture**, inspired by adapters in NLP.
- **Three tailored modules** supplying vision-specific inductive biases:
  1. **Spatial prior module** — captures local semantics (spatial prior) from the input images.
  2. **Spatial feature injector** — incorporates the spatial prior into the ViT.
  3. **Multi-scale feature extractor** — reconstructs the multi-scale features dense prediction requires.
- **Flexible transfer paradigm** — rather than the usual large-scale image pre-training then fine-tuning, the backbone is a general-purpose model (pre-trainable on multi-modal data) and a **randomly initialized adapter** introduces image-related priors at transfer time.
- **Matches or beats vision-specific transformers** — using ViT as backbone, achieves comparable or better performance than Swin.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Chen_et_al._2022_2205.08534.md`](references/papers/Chen_et_al._2022_2205.08534.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (plain ViT, NLP adapters, Swin → ViT-Adapter; siblings: PVT, Swin V2, MaxViT; contrast: vision-specific transformers, which bake in image priors but cannot exploit multi-modal pre-training).
