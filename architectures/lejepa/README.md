# LeJEPA

## Overview

LeJEPA is a **provable and scalable** joint-embedding predictive architecture that eliminates all the heuristics of prior JEPA methods — no stop-gradient, no EMA teacher, no predictor network, no structured masking. Its key innovation is **SIGReg** (Sketched Isotropic Gaussian Regularization), which constrains the embedding distribution to match an isotropic Gaussian via the Cramér-Wold theorem, reducing the distribution-matching problem to a set of one-dimensional normality tests along random directions. The theoretical contribution: an isotropic Gaussian is the **optimal** embedding distribution for minimizing downstream prediction risk.

- **Year:** 2025
- **Authors:** Randall Balestriero, Yann LeCun (Meta FAIR / Brown University)
- **Source:** arXiv:2511.08544 — *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*
- **Category:** DL/Self-Supervised Representation

## Key Characteristics

- **SIGReg replaces all heuristics**: no EMA target encoder, no stop-gradient, no predictor, no structured masking — just a single encoder + small projector + distribution regularization.
- **Isotropic Gaussian is provably optimal**: the theoretical finding that Gaussian distribution minimizes downstream prediction risk under the JEPA alignment objective.
- **Cramér-Wold sketching**: high-dimensional distribution matching is reduced to linear-complexity univariate normality tests along random projections — ~50 lines of code.
- **Stable across 60+ architectures and 10+ datasets**: no hyperparameter tuning needed; the same configuration works everywhere.
- **ImageNet-1K**: ViT-H/14 achieves 79% linear probing accuracy.
- **Predecessor to LeWM, LeVJEPA, LpWM**: SIGReg becomes the standard collapse-prevention mechanism for the entire 2026 JEPA line.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Balestriero_LeCun_2025_2511.08544.md`](references/papers/Balestriero_LeCun_2025_2511.08544.md)

## Related Architectures

See `architecture.md` → Evolution section (VICReg → LeJEPA; siblings: I-JEPA, V-JEPA; successors: LeWM, LeVJEPA, LpWM, AD-E2E-JEPA all adopt SIGReg).
