# VMamba

## Overview

Vision backbones face a choice: CNNs are cheap but limited, ViTs learn well at scale but self-attention is **quadratic in token count**, which hurts badly at the large spatial resolutions dense prediction needs. Existing efficient-attention compromises either shrink the effective receptive field or degrade across tasks. VMamba ports Mamba — a state-space model with linear complexity — into a vision backbone, with the key question being how a 1D ordered scan can serve non-sequential 2D data.

- **Year:** 2024
- **Authors:** Liu et al. (University of Science and Technology of China, BAAI, University of Macau, Institute of Automation CAS)
- **Source:** arXiv:2401.10166 — *VMamba: Visual State Space Model*
- **Category:** DL/Vision SSM

## Key Characteristics

- **Linear-time vision backbone** — Mamba's state-space formulation adapted to images.
- **Visual State-Space (VSS) block** — the building block, built around the 2D Selective Scan module.
- **2D Selective Scan (SS2D)** — traverses **four scanning routes** to bridge the ordered 1D selective scan and the non-sequential structure of 2D data, collecting context from multiple sources and perspectives.
- **Global receptive field and dynamic weights, kept** — the design target is to preserve vanilla self-attention's advantages while avoiding quadratic cost.
- **Superior input scaling efficiency** — highlighted as the main practical gain versus benchmark models, i.e. cost grows much more slowly as resolution rises.
- **Preserves linear complexity** while performing well across diverse visual perception tasks; architectural and implementation enhancements are applied on top.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2024_2401.10166.md`](references/papers/Liu_et_al._2024_2401.10166.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Mamba (NLP SSM) → VMamba; siblings: MambaVision (hybrid Mamba-Transformer), Vim, LocalMamba; contrast: ViT with quadratic attention, CNN backbones).
