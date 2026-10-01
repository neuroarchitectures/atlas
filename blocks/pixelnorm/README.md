# Pixel Normalization (GAN)

## Design Philosophy

Normalize each pixel's feature vector to unit length. Prevents magnitude explosion in GAN generators, which is common with transposed convolutions.

## Functionality

For each spatial location (h, w), normalize across channels: x_norm = x / sqrt(mean_c(x_c^2) + epsilon). Unlike BatchNorm, this is per-pixel, not per-batch.

## Used By

Progressive GAN | StyleGAN (early versions)

## Features

- **No batch statistics**: Per-pixel normalization.
- **Prevents explosion**: Keeps feature magnitudes bounded.
- **No running averages**: Simpler than BatchNorm.

## Evolution

Predecessor: BatchNorm. Successor: AdaIN (StyleGAN), weight demodulation (StyleGAN2).
