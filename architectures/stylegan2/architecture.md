# Architecture: StyleGAN2

## Motivation

Improved StyleGAN with weight demodulation and path length regularization for high-quality face generation.

## Core Idea

Improved StyleGAN with weight demodulation and path length regularization for high-quality face generation.

## Architecture

### Overview

![stylegan2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent Z | `input` | shape=[512] |
| 2 | Mapping Network (8 MLP) | `linear` | outFeatures=512 |
| 3 | Synthesis Network (mod-based) | `conv2d` | outFeatures=3 |
| 4 | Generated Image | `output` | — |

</details>

### Components

1. **Latent Z** — input layer. 2. **Mapping Network (8 MLP)** — linear layer. 2. **Synthesis Network (mod-based)** — conv2d layer. 2. **Generated Image** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Improved StyleGAN with weight demodulation and path length regularization for high-quality face generation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
