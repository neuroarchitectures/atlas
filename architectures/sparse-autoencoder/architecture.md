# Architecture: Sparse Autoencoder

## Motivation

Standard autoencoders can learn trivial identity mappings when the latent dimension exceeds the input dimension. Sparsity constraints force the network to learn meaningful features even in overcomplete settings.

## Core Idea

Add an L1 penalty or KL-divergence penalty on the latent activations to encourage most units to be inactive for any given input, creating a sparse distributed representation.

## Architecture

### Overview

![sparse autoencoder architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [784] |
| 2 | Encoder | `linear` | outFeatures: 256, inFeatures: 784 |
| 3 | ReLU+Sparsity | `relu` | L1 penalty on activations |
| 4 | Sparse Latent | `identity` |  |
| 5 | Decoder | `linear` | outFeatures: 784, inFeatures: 256 |
| 6 | Reconstruction | `output` |  |

</details>

### Components

1. **Encoder** — Maps input to a higher-dimensional latent space. 2. **Sparsity penalty** — L1 regularization or KL-divergence on the average activation forces most latent units to be near zero. 3. **Decoder** — Reconstructs the input from the sparse latent representation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Add an L1 penalty or KL-divergence penalty on the latent activations to encourage most units to be inactive for any given input, creating a sparse distributed representation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Autoencoder. Successor: Denoising Autoencoder, Contractive Autoencoder.

## References

- Ranzato et al. 2007
