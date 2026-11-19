# LpWM: A Case for Sparse Representations in World Models

**Authors:** Linqi Kuang, Randall Balestriero, Yann LeCun
**arXiv:** 2608.22764 (2026)

## Abstract

JEPA world models typically use dense continuous representations. We challenge this assumption, showing that for Lipschitz controlled dynamics, sufficiently high-dimensional sparse (one-hot) representations enable action-conditioned linear dynamics to approximate nonlinear dynamics with rollout error vanishing as dimensionality grows (Proposition 1, Corollary 1). We introduce LpWM, which uses RDMReg (Rectified Distribution Matching) to produce non-negative sparse codes by matching features to a rectified generalized Gaussian distribution. On PushT, sparse LpWM improves planning success by up to 57% over dense LeWM with medium-capacity predictors. We show that sparse codes are mode-decomposable: the active support encodes discrete dynamics regimes while feature magnitudes capture continuous within-regime state, providing a natural representation for mixed discrete-continuous physical dynamics.
