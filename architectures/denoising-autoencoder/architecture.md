# Architecture: Denoising Autoencoder

## Motivation

Standard autoencoders risk learning the identity function. By corrupting the input and training to reconstruct the clean version, the model is forced to learn the structure of the data manifold rather than trivial copying.

## Core Idea

Add noise (Gaussian, masking, or salt-and-pepper) to the input, then train the autoencoder to minimize the reconstruction loss between the output and the original (uncorrupted) input.

## Architecture

### Overview

![denoising autoencoder architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [784] |
| 2 | Noise Injection | `identity` | Gaussian/masking noise |
| 3 | Encoder | `linear` | outFeatures: 128, inFeatures: 784 |
| 4 | ReLU | `relu` |  |
| 5 | Decoder | `linear` | outFeatures: 784, inFeatures: 128 |
| 6 | Clean Reconstruction | `output` | loss vs. clean input |

</details>

### Components

1. **Noise injection** — Corrupts the input with stochastic noise (Gaussian, masking, or salt-and-pepper). 2. **Encoder-decoder** — Standard autoencoder that maps corrupted input to a latent representation and back. 3. **Cross-entropy or MSE loss** — Computed against the original clean input, not the corrupted version.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Add noise (Gaussian, masking, or salt-and-pepper) to the input, then train the autoencoder to minimize the reconstruction loss between the output and the original (uncorrupted) input.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Sparse Autoencoder. Successor: Contractive Autoencoder, Stacked Denoising Autoencoders (SDA).

## References

- Vincent et al. 2008
