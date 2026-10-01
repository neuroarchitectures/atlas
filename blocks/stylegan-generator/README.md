# StyleGAN Generator Block

## Design Philosophy

Instead of feeding noise at the input, use a learned constant and inject style at each layer via AdaIN. This decouples the noise (global structure) from style (local appearance).

## Functionality

1. Learned constant (4x4x512). 2. Upsample (transposed conv). 3. AdaIN: normalize, then modulate with style w (scale + bias from w). 4. Conv. 5. Noise injection (per-pixel Gaussian).

## Used By

StyleGAN | StyleGAN2 | StyleGAN3

## Features

- **Style-based**: Style vector w controls appearance at each layer.
- **Noise injection**: Stochastic detail without affecting style.
- **Hierarchical**: Coarse-to-fine style control (resolution-dependent).

## Evolution

Predecessor: Progressive GAN. Successor: StyleGAN2 (replace AdaIN with weight demodulation), StyleGAN3 (alias-free).
