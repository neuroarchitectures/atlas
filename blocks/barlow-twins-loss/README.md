# Barlow Twins Loss

## Design Philosophy

Self-supervised learning without negatives. Compute the cross-correlation matrix between two augmented views and push it toward the identity matrix (decorrelation + invariance).

## Functionality

L = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2, where C is the cross-correlation matrix. The first term makes representations invariant (diagonal = 1), the second makes them decorrelated (off-diagonal = 0).

## Used By

Barlow Twins (self-supervised) | VICReg | Medical imaging SSL

## Features

- **No negatives needed**: Avoids collapse without negative samples.
- **Decorrelation**: Prevents redundant features.
- **Simple**: Just a correlation matrix and MSE.

## Evolution

Predecessor: SimCLR, BYOL. Successor: VICReg, SwAV variants.
