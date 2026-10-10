# VICReg Loss

## Design Philosophy

Variance, Invariance, Covariance Regularization. Combine three terms: variance (maintain per-dimension std), invariance (pull views together), covariance (decorrelate dimensions).

## Functionality

L = lambda * Var(z) + mu * Inv(z1, z2) + nu * Cov(z). Var: std(z) per dimension, clamped at 1. Inv: MSE between views. Cov: off-diagonal covariance near 0.

## Used By

VICReg (self-supervised) | Multi-modal SSL | Video SSL

## Features

- **No negatives**: Like BYOL, avoids collapse via variance + covariance terms.
- **Explicit regularization**: Each term has a clear purpose.
- **Batch-level**: Works with any batch size.

## Evolution

Predecessor: Barlow Twins, BYOL. Successor: VICReg variants, MoCo v3.
