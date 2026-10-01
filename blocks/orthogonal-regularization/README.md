# Orthogonal Regularization

## Design Philosophy

Encourage weight matrices to be orthogonal, which prevents singular values from collapsing to zero and improves training stability, especially in GANs.

## Functionality

L_reg = ||W W^T - I||_F^2 (Frobenius norm). Soft constraint added to the main loss.

## Used By

BigGAN | GAN stabilization | Representation learning

## Features

- **Prevents collapse**: Orthogonal weights maintain gradient flow.
- **Soft constraint**: Added as a regularizer, not hard constraint.
- **Simple to implement**: Matrix multiplication + norm.

## Evolution

Predecessor: Spectral normalization. Successor: Hard orthogonal constraints via QR decomposition.
