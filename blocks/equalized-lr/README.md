# Equalized Learning Rate (GAN)

## Design Philosophy

Scale weight initializations so that all layers have the same learning dynamics. This ensures that deeper layers don't train slower than shallow ones, which is critical for GAN stability.

## Functionality

Initialize weights from N(0, 1), then scale by a layer-dependent constant c = sqrt(2 / fan_in) at runtime. This is equivalent to He init but applied dynamically, not at initialization.

## Used By

Progressive GAN | StyleGAN

## Features

- **Equalized dynamics**: All layers learn at the same rate.
- **Runtime scaling**: Applied at each forward pass, not just init.
- **Simple**: Just multiply by a constant.

## Evolution

Predecessor: He initialization. Successor: LR-Equalized variants in modern GANs.
