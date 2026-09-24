# Vision Mamba

## Overview

SSMs with efficient hardware-aware designs — Mamba — had shown great potential for long sequence modelling. Building efficient and generic vision backbones **purely upon SSMs** is therefore an appealing direction, but visual data is hard for SSMs: it is **position-sensitive** and visual understanding requires **global context**. Vision Mamba (Vim) shows that the reliance on self-attention for visual representation learning is **not necessary**, and beats established vision transformers while being substantially cheaper.

- **Year:** 2024
- **Authors:** Zhu et al. (Huazhong University of Science and Technology, University of Adelaide, Wuhan AI Research, University of Chinese Academy of Sciences)
- **Source:** arXiv:2401.09417 — *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*
- **Category:** DL/Vision SSM

## Key Characteristics

- **Reliance on self-attention is not necessary** — the central claim; a generic vision backbone is built purely on SSMs.
- **Bidirectional Mamba blocks (Vim)** — image sequences are **marked with position embeddings** and the visual representation is **compressed with bidirectional state space models**; bidirectionality supplies the global context that unidirectional SSMs lack.
- **Position-sensitivity handled explicitly** — position embeddings are the fix for the stated SSM weakness on visual data.
- **Linear complexity at high resolution** — **2.8× faster than DeiT** and **86.8% GPU memory saved** when performing batch inference to extract features on images at **1248×1248**.
- **Better than DeiT across three tasks** — higher performance on ImageNet classification, COCO object detection and ADE20K semantic segmentation.
- **Positioned as a next-generation backbone** — the paper states Vim has great potential to be the backbone for vision foundation models, overcoming Transformer-style understanding's compute and memory constraints on high-resolution images.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhu_et_al._2024_2401.09417.md`](references/papers/Zhu_et_al._2024_2401.09417.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (S4/LSSL/DSS/S4D, Mamba → Vim; siblings: LocalMamba, VMamba, MambaVision; contrast: DeiT and Vision Transformers, which Vim outperforms while being faster and lighter).
