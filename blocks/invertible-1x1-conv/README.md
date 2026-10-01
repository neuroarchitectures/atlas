# Invertible 1x1 Convolution (Glow)

## Design Philosophy

A 1x1 convolution is a linear transformation of channels. Making it invertible allows it to be used as a learnable permutation in normalizing flows, replacing fixed channel permutations.

## Functionality

W is a c x c weight matrix. Forward: y = W * x. Inverse: x = W^{-1} * y. Log-determinant: log|det(W)| is computed directly. Initialized as a random rotation matrix.

## Used By

Glow | Flow-based models | Invertible architectures

## Features

- **Learnable permutation**: Replaces fixed channel shuffling.
- **Exact log-determinant**: O(c^3) but c is small.
- **LU decomposition**: For efficient inverse computation.

## Evolution

Predecessor: Fixed permutations (RealNVP). Successor: Emerging (convex potential flows).
