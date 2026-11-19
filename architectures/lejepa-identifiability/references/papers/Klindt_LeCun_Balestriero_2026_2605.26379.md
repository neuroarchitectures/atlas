# When Does LeJEPA Learn a World Model?

**Authors:** David Klindt, Yann LeCun, Randall Balestriero
**arXiv:** 2605.26379 (2026, NeurIPS 2026)

## Abstract

We prove that LeJEPA — a joint-embedding predictive architecture with alignment loss and SIGReg (isotropic Gaussian regularization) — linearly identifies the latent variables of a class of worlds where latent variables evolve under stationary additive noise transitions. "Linearly identifies" means the learned representation recovers the true latent variables up to an orthogonal transformation, from nonlinear observations. We further prove that the Gaussian distribution is the unique distribution for which this identifiability guarantee holds, providing the strongest possible theoretical justification for SIGReg. We establish approximate identifiability for cases where conditions are partially violated, showing graceful degradation. Finally, we show that linear orthogonal identifiability enables optimal latent-space planning. We validate our results from 2D examples to 4096-dimensional latent spaces, including ablations, pixel-based robot control, and ImageNet-X natural images.
