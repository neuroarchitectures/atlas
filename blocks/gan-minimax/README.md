# GAN Minimax Block (Generator + Discriminator)

## Design Philosophy

Two networks trained in a **minimax game**: the generator tries to create realistic samples to deceive the discriminator; the discriminator tries to distinguish real from fake. The philosophy: instead of specifying a likelihood or distance metric, pit two networks against each other — the discriminator's learned ability to distinguish provides the gradient for the generator. At the optimum, this minimizes Jensen-Shannon divergence.

## Functionality

- **Generator `G(z, θ_g)`**: Maps noise `z ~ p_z` to data space. Any differentiable architecture (transposed convs for images, MLPs, etc.).
- **Discriminator `D(x, θ_d)`**: Outputs a scalar probability that `x` is real.
- **Objective**: `min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]`.
- **Non-saturating generator loss**: `max_G log(D(G(z)))` — stronger gradients early in training.
- **Alternating updates**: G and D updated alternately; terminate at Nash equilibrium.

## Used By

| Model | Role |
|-------|------|
| GAN (Goodfellow, 2014) | The defining framework |
| DCGAN | Convolutional generator/discriminator |
| StyleGAN | Adaptive instance normalization for high-quality faces |
| BigGAN | Large-scale, truncation tricks |
| CycleGAN | Unpaired image-to-image translation |
| MolGAN, GraphGAN | Graph-structured generation |

## Features

- **Parallelizable generation**: Entire sample in one forward pass (unlike autoregressive).
- **Flexible generator**: Any differentiable architecture.
- **Implicit density**: No explicit `p(x)` — sample quality, not likelihood.
- **Mode collapse / instability**: The key weaknesses, addressed by WGAN, SN-GAN, etc.

## Evolution

- **Predecessor**: Predictability minimization; explicit density models (VAE, Boltzmann).
- **Successor**: WGAN (Wasserstein distance); StyleGAN; ultimately diffusion models (DDPM, 2020) surpassed GANs in sample quality.
