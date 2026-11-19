# LeVJEPA

## Overview

LeVJEPA represents the state of the art in JEPA efficiency optimization. By using SIGReg as the sole collapse-prevention mechanism, it completely eliminates the EMA target encoder, the predictor network, stop-gradient, and structured masking. The architecture reduces to: **single encoder + small projector + SIGReg**. Despite this radical simplification, LeVJEPA matches or exceeds V-JEPA 2's performance with **5.6–20.8× less compute**, and exceeds the strongest video baseline by 7.6 points on ImageNet-1K at equal FLOP budget. A ViT-Tiny variant trains to non-trivial ImageNet accuracy in 12 hours on a single consumer GPU.

- **Year:** 2026
- **Authors:** Julian Kuhn, Florian Maes, Jonathan Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Jan Buettner (Meta FAIR / TU Munich / Inria)
- **Source:** arXiv:2608.27395 — *LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics*
- **Category:** DL/Self-Supervised Representation

## Key Characteristics

- **No EMA target encoder, no predictor, no stop-gradient, no structured masking** — SIGReg alone prevents collapse.
- **5.6–20.8× less compute** than V-JEPA 2 at matching or better performance (ViT-S/B/L).
- **+7.6 points on ImageNet-1K** over strongest video baseline at equal total FLOP budget.
- **Block-causal attention** — frame representations computed from past observations only (no re-encoding), enabling autoregressive world models and streaming inference.
- **Random token dropping (up to 95%)** replaces structured masking — acts as augmentation, not approximation (33.9% → 47.6% improvement at 95% drop).
- **ViT-Tiny: 12 hours, single consumer GPU** → non-trivial ImageNet accuracy.
- **Time ordering emerges in the encoder itself** — block-causal attention performs comparably to full bidirectional attention in frozen probing.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Kuhn_et_al._2026_2608.27395.md`](references/papers/Kuhn_et_al._2026_2608.27395.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA 2 → LeVJEPA; sibling: LeJEPA provides SIGReg; contrast: V-JEPA 2 uses EMA + predictor + structured masking).
