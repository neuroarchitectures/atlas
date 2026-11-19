# LpWM

## Overview

LpWM (Latent sparse World Model) challenges a fundamental assumption in JEPA world models: that **dense representations** are the optimal geometry for modeling dynamics. LpWM's theoretical motivation shows that for Lipschitz controlled dynamics, sufficiently high-dimensional **one-hot (sparse) representations** can approximate nonlinear dynamics with action-conditioned linear dynamics, with rollout error vanishing as dimensionality grows. Using RDMReg (Rectified Distribution Matching) to produce non-negative sparse codes, LpWM achieves up to **57% improvement in planning success** over dense LeWM on PushT, and reveals that sparse representations are **mode-decomposable**: the support encodes discrete dynamics regimes while feature magnitudes capture continuous within-regime state.

- **Year:** 2026
- **Authors:** Linqi Kuang, Randall Balestriero, Yann LeCun (Meta FAIR / NYU)
- **Source:** arXiv:2608.22764 — *LpWM: A Case for Sparse Representations in World Models*
- **Category:** DL/World-Model

## Key Characteristics

- **Sparse representations over dense**: challenges the default assumption that dense embeddings are optimal for world models.
- **Theoretical foundation**: for Lipschitz controlled dynamics, one-hot representations enable action-conditioned linear dynamics approximation with vanishing rollout error (Proposition 1, Corollary 1).
- **RDMReg regularization**: matches features to a rectified generalized Gaussian distribution, producing non-negative sparse codes.
- **Mode decomposition**: sparse codes decompose into modes — support encodes discrete dynamics regime (e.g., contact/no-contact), magnitudes encode continuous within-regime state.
- **+57% planning success** on PushT over dense LeWM (with medium-capacity predictor).
- **Complementary to LeHami**: LeHami structures dynamics via Hamiltonian (q,p) phase space; LpWM structures via sparsity and mode decomposition.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Kuang_et_al._2026_2608.22764.md`](references/papers/Kuang_et_al._2026_2608.22764.md)

## Related Architectures

See `architecture.md` → Evolution section (LeWM → LpWM; sibling: LeHami provides symplectic dynamics; contrast: LeWM uses dense representations).
