# AD-E2E-JEPA

## Overview

AD-E2E-JEPA applies the JEPA world model framework to **end-to-end autonomous driving**. By introducing a SIGReg-regularized learnable projector that reduces planning patch count by 16× and embedding dimension by 4×, it achieves **100× inference acceleration** (0.8 seconds for 8-frame rollout over 256 candidate trajectories). The world model itself performs zero-shot goal-conditioned planning — **no trained driving policy** is needed. On NAVSIMv2, it achieves 67.3/72.9 EPDMS (with multiplicative safety metric) and 84.1/86.5 EPDMS† (without).

- **Year:** 2026
- **Authors:** Zhiwu Zhu, Qingwen Zhang, Yann LeCun, Anna Choromanska (Meta FAIR / ASU)
- **Source:** arXiv:2609.34085 — *AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving*
- **Category:** DL/Autonomous-Driving

## Key Characteristics

- **JEPA world model for driving**: latent-space prediction of future driving scenes conditioned on candidate trajectories.
- **SIGReg projector**: reduces planning patches by 16× and embedding dim by 4×, enabling 100× faster inference.
- **100× inference acceleration**: 0.8s for 8-frame rollout / 256 candidate trajectories.
- **No trained driving policy**: the world model itself performs zero-shot goal-conditioned planning by scoring trajectory candidates in latent space.
- **NAVSIMv2: 67.3/72.9 EPDMS** (with safety metric), **84.1/86.5 EPDMS†** (without).
- **Self-supervised pretraining projector** improves downstream imitation learning from 80.2 to 85.4 EPDMS.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhu_et_al._2026_2609.34085.md`](references/papers/Zhu_et_al._2026_2609.34085.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA 2-AC → AD-E2E-JEPA; sibling: LeWM provides end-to-end training; the SIGReg projector design is transferable to any JEPA planning system).
