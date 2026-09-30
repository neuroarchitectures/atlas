# Architecture: Autoencoder

## Motivation

How can a neural network learn useful representations without labels? The autoencoder addresses this by training the network to reconstruct its own input through a bottleneck, forcing it to discover the most important structure in the data.

## Core Idea

An encoder maps input x to a lower-dimensional latent representation z = f(x), and a decoder maps z back to a reconstruction x' = g(z). The network is trained to minimize reconstruction loss ||x - x'||. The bottleneck dimensionality forces the network to learn a compressed representation that captures the most important features.

## Architecture

### Overview

![autoencoder architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input x | `input` | shape: [1, 784] |
| 2 | Encoder L1 | `linear` | outFeatures: 512, inFeatures: 784 |
| 3 | ReLU | `relu` |  |
| 4 | Encoder L2 | `linear` | outFeatures: 256, inFeatures: 512 |
| 5 | ReLU | `relu` |  |
| 6 | Bottleneck z | `linear` | outFeatures: 32, inFeatures: 256 |
| 7 | Decoder L1 | `linear` | outFeatures: 256, inFeatures: 32 |
| 8 | ReLU | `relu` |  |
| 9 | Decoder L2 | `linear` | outFeatures: 512, inFeatures: 256 |
| 10 | ReLU | `relu` |  |
| 11 | Reconstruction | `linear` | outFeatures: 784, inFeatures: 512 |
| 12 | x' | `output` |  |

</details>

An encoder maps input x to a lower-dimensional latent representation z = f(x), and a decoder maps z back to a reconstruction x' = g(z). The network is trained to minimize reconstruction loss ||x - x'||. The bottleneck dimensionality forces the network to learn a compressed representation that captures the most important features.

### Components

2. **Encoder L1** (`linear`, scope: `encoder`) — Params: outFeatures: 512, inFeatures: 784
3. **ReLU** (`relu`, scope: `encoder`) — Params: none
4. **Encoder L2** (`linear`, scope: `encoder`) — Params: outFeatures: 256, inFeatures: 512
5. **ReLU** (`relu`, scope: `encoder`) — Params: none
6. **Bottleneck z** (`linear`, scope: `encoder`) — Params: outFeatures: 32, inFeatures: 256
7. **Decoder L1** (`linear`, scope: `decoder`) — Params: outFeatures: 256, inFeatures: 32
8. **ReLU** (`relu`, scope: `decoder`) — Params: none
9. **Decoder L2** (`linear`, scope: `decoder`) — Params: outFeatures: 512, inFeatures: 256
10. **ReLU** (`relu`, scope: `decoder`) — Params: none
11. **Reconstruction** (`linear`, scope: `decoder`) — Params: outFeatures: 784, inFeatures: 512

### Data Flow

The architecture processes input through a sequence of 12 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Autoencoder.

## Evolution

The Autoencoder introduced the encoder-decoder bottleneck for representation learning. Successors: Denoising AE, Sparse AE, Contractive AE, VAE (probabilistic), β-VAE (disentangled), and modern self-supervised models (MAE, SimMIM). The encoder-decoder pattern underpins U-Net, seq2seq, and VAE-based diffusion.

## Source

- **Paper:** Parallel Distributed Processing 1986
- **Year:** 1986
- **Authors:** Rumelhart et al.
