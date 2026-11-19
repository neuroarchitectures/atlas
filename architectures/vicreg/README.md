# VICReg

## Overview

VICReg (Variance-Invariance-Covariance Regularization) is a self-supervised learning method that prevents representation collapse using **three explicit regularization terms** on the embedding distribution — no architectural asymmetry required (no stop-gradient, no EMA teacher, no memory bank, no large batch of negatives). It was the first in the LeCun line to show that **explicit distribution constraints** can replace the engineering tricks of contrastive and non-contrastive SSL.

- **Year:** 2021 (ICLR 2022)
- **Authors:** Adrien Bardes, Jean Ponce, Yann LeCun (Meta AI / NYU)
- **Source:** arXiv:2105.04906 — *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*
- **Category:** DL/Self-Supervised Representation

## Key Characteristics

- **Three-term loss**: (1) **Invariance** — same image's two views produce similar embeddings; (2) **Variance** — each embedding dimension maintains variance above a threshold; (3) **Covariance** — off-diagonal covariance is driven to zero, decorrelating dimensions.
- **No architectural asymmetry**: unlike BYOL/SimSiam (stop-gradient + EMA) or SimCLR (negatives), VICReg works with a symmetric twin-encoder setup.
- **Variance term**: a hinge loss `max(0, γ − std(z))` per dimension, keeping every dimension active.
- **Covariance term**: the off-diagonal entries of the batch covariance matrix are penalized → prevents redundant dimensions and rank collapse.
- **Information-theoretic analysis** (Shwartz-Ziv et al., 2023) showed VICReg's objective is equivalent to mutual information optimization with provable generalization bounds.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Bardes_et_al._2021_2105.04906.md`](references/papers/Bardes_et_al._2021_2105.04906.md)

## Related Architectures

See `architecture.md` → Evolution section (Barlow Twins → VICReg → SIGReg/LeJEPA; siblings: SimCLR, BYOL, SimSiam; successor: LeJEPA which replaces second-order statistics with full distribution matching).
