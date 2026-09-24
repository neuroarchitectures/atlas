# I-JEPA

## Overview

Self-supervised image learning splits into invariance-based methods (which need hand-crafted augmentations and bake in their biases) and generative methods (which reconstruct pixels). I-JEPA is a third option: **non-generative and augmentation-light** — predict the *representations* of target blocks from a context block in the same image, with the masking strategy carrying the burden that augmentations normally carry.

- **Year:** 2023
- **Authors:** Assran et al. (Meta AI - FAIR, INRIA, Sorbonne Université, MPI for Intelligent Systems, NEC Labs Europe, Université Paris-Dauphine, Mila Québec)
- **Source:** arXiv:2301.08243 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*
- **Category:** DL/Self-Supervised Representation

## Key Characteristics

- **Predict in representation space** — from a single context block, predict the representations of various target blocks of the same image; no pixel reconstruction, no hand-crafted augmentation reliance.
- **Masking strategy is the core design choice** — it must (a) sample target blocks with **sufficiently large scale** (semantic), and (b) use a **sufficiently informative, spatially distributed** context block.
- **Avoids the invariance bias problem** — invariance-based pretraining introduces biases that may hurt downstream tasks requiring different abstraction levels, and image-specific augmentations do not transfer to modalities like audio.
- **Highly scalable with ViTs** — a ViT-Huge/14 on ImageNet trains on 16 A100 GPUs in under 72 hours, with strong downstream performance from linear classification to object counting and depth prediction.
- **Broad transfer** — evaluated across a wide range of tasks, not just classification.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Assran_et_al._2023_2301.08243.md`](references/papers/Assran_et_al._2023_2301.08243.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (invariance-based SSL and MAE → I-JEPA; siblings: V-JEPA, V-JEPA 2 (video extension, in catalog), VideoMAE; contrast: generative masked autoencoders that reconstruct pixels).
