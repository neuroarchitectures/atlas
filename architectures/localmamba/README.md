# LocalMamba

## Overview

SSMs, notably Mamba, had progressed on long-sequence modelling in language, but their application in vision had **not markedly surpassed traditional CNNs and ViTs**. LocalMamba's diagnosis is concrete: the key to enhancing Vision Mamba lies in **optimizing scan directions**, because traditional ViM approaches **flatten spatial tokens**, overlooking the preservation of **local 2D dependencies** and thereby **elongating the distance between adjacent tokens**.

- **Year:** 2024
- **Authors:** Huang et al. (Australian National University, ShanghaiTech University)
- **Source:** arXiv:2403.09338 — *LocalMamba: Visual State Space Model with Windowed Selective Scan*
- **Category:** DL/Vision SSM

## Key Characteristics

1. **Windowed local scanning** — divides images into distinct windows, effectively capturing local dependencies **while maintaining a global perspective**; the direct fix for distance elongation from flattening.
2. **Dynamic layer-wise scan search** — acknowledging that **different network layers have varying preferences for scan patterns**, a dynamic method **independently searches for the optimal scan choices for each layer**, substantially improving performance.
3. **Superiority across plain and hierarchical models** — extensive experiments on both underline the approach's effectiveness at capturing image representations.
4. **Concrete gain over Vim** — **outperforms Vim-Ti by 3.1% on ImageNet with the same 1.5G FLOPs**.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Huang_et_al._2024_2403.09338.md`](references/papers/Huang_et_al._2024_2403.09338.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (S4, Mamba, Vim → LocalMamba; siblings: VMamba, MambaVision, Vision Mamba; contrast: Vim/Ti, the +3.1% comparison point at equal FLOPs, and CNN/ViT baselines that plain ViM had failed to clearly surpass).
