# Architecture: VAE

## Motivation

Standard autoencoders learn deterministic mappings but cannot generate new samples — the latent space is not structured. The VAE regularizes the latent space to follow a known distribution (Gaussian), enabling generation by sampling from the prior and decoding.

## Core Idea

The encoder outputs a distribution (mean μ and variance σ²) rather than a point. The reparameterization trick (z = μ + σ·ε, ε~N(0,1)) allows gradients to flow through the sampling step. The loss combines reconstruction loss and KL divergence regularization, ensuring the latent distribution stays close to N(0,1).

## Architecture

### Overview

![vae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input x | `input` | shape: [1, 784] |
| 2 | Encoder L1 | `linear` | outFeatures: 512, inFeatures: 784 |
| 3 | ReLU | `relu` |  |
| 4 | μ | `linear` | outFeatures: 32, inFeatures: 512 |
| 5 | log σ² | `linear` | outFeatures: 32, inFeatures: 512 |
| 6 | Reparameterize | `custom` | type: reparameterization |
| 7 | KL Divergence | `custom` | type: kl_divergence |
| 8 | Decoder L1 | `linear` | outFeatures: 512, inFeatures: 32 |
| 9 | ReLU | `relu` |  |
| 10 | Decoder L2 | `linear` | outFeatures: 784, inFeatures: 512 |
| 11 | Reconstruction x' | `output` |  |
| 12 | ELBO Loss | `loss` | type: elbo |

</details>

The encoder outputs a distribution (mean μ and variance σ²) rather than a point. The reparameterization trick (z = μ + σ·ε, ε~N(0,1)) allows gradients to flow through the sampling step. The loss combines reconstruction loss and KL divergence regularization, ensuring the latent distribution stays close to N(0,1).

### Components

2. **Encoder L1** (`linear`, scope: `encoder`) — Params: outFeatures: 512, inFeatures: 784
3. **ReLU** (`relu`, scope: `encoder`) — Params: none
4. **μ** (`linear`, scope: `encoder`) — Params: outFeatures: 32, inFeatures: 512
5. **log σ²** (`linear`, scope: `encoder`) — Params: outFeatures: 32, inFeatures: 512
6. **Reparameterize** (`custom`, scope: `encoder`) — Params: type: reparameterization
7. **KL Divergence** (`custom`, scope: `loss`) — Params: type: kl_divergence
8. **Decoder L1** (`linear`, scope: `decoder`) — Params: outFeatures: 512, inFeatures: 32
9. **ReLU** (`relu`, scope: `decoder`) — Params: none
10. **Decoder L2** (`linear`, scope: `decoder`) — Params: outFeatures: 784, inFeatures: 512
12. **ELBO Loss** (`loss`, scope: `loss`) — Params: type: elbo

### Data Flow

The architecture processes input through a sequence of 12 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to VAE.

## Evolution

VAE introduced probabilistic latent variable models with differentiable sampling. Successors: β-VAE (disentanglement), VQ-VAE (discrete latents), NVAE (hierarchical), and VAE-VAE (dual autoencoders). The VAE encoder is also used in Latent Diffusion (Stable Diffusion) to compress images to latent space.

## Source

- **Paper:** arXiv:1312.6114
- **Year:** 2013
- **Authors:** Kingma & Welling
