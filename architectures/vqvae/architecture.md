# Architecture: VQ-VAE

## Motivation

Vector Quantized VAE: discrete latent space with codebook learning for high-fidelity generation.

## Core Idea

Vector Quantized VAE: discrete latent space with codebook learning for high-fidelity generation.

## Architecture

### Overview

![vqvae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 128, 128] |
| 2 | CNN Encoder | `conv2d` | outFeatures=256 |
| 3 | Vector Quantization (Codebook) | `embedding` | vocabSize=512, dim=256 |
| 4 | CNN Decoder | `conv2d` | outFeatures=3 |
| 5 | Reconstruction | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **CNN Encoder** — conv2d layer. 2. **Vector Quantization (Codebook)** — embedding layer. 2. **CNN Decoder** — conv2d layer. 2. **Reconstruction** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Vector Quantized VAE: discrete latent space with codebook learning for high-fidelity generation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
