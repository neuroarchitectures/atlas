# Mixture-of-Laplace Loss

## Design Philosophy

L1 on flow errors implicitly assumes a single Laplace distribution — but real flow error is heavy-tailed and multimodal. SEA-RAFT models the error as a *mixture of Laplace distributions*, predicting per-component scales and mixing weights: robust to outliers, expressive for ambiguous motion.

## Functionality

- Predict K Laplace components (locations = regression output, scales + weights learned); negative log-likelihood of the mixture replaces L1 at every refinement iteration.

## Used By

| Model | Role |
|-------|------|
| SEA-RAFT | Supervises every ConvGRU refinement iteration |

## Features

- **Heavy-tail robustness** with learned adaptive scale.
- **Uncertainty estimate** falls out of the mixture.

## Evolution

- **Predecessor**: L1/L2, Laplace NLL, per-pixel uncertainty weighting (confidence-weighted-loss family).
- **Related**: probabilistic flow losses (BaM-L); distributional regression heads.
