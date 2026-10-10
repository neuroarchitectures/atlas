# Spectral Normalization

## Design Philosophy

Normalize weight matrices by their spectral norm (largest singular value) to constrain the Lipschitz constant of the network, stabilizing GAN training.

## Functionality

W_normalized = W / sigma(W), where sigma(W) is the largest singular value. Computed efficiently via power iteration (one step per forward pass).

## Used By

SAGAN, BigGAN | GAN stabilization | Critic regularization in WGAN-GP

## Features

- **Lipschitz constraint**: Bounds the gradient norm, stabilizing training.
- **Power iteration**: O(1) per step, negligible overhead.
- **Compatible**: Can be applied to any layer.

## Evolution

Predecessor: Weight normalization, gradient penalty. Successor: Conditional spectral norm (conditional GANs).
