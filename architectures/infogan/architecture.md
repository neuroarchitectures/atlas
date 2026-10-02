# Architecture: InfoGAN

## Motivation

GAN with disentangled latent codes: maximizes mutual information between latent codes and generated data.

## Core Idea

GAN with disentangled latent codes: maximizes mutual information between latent codes and generated data.

## Architecture

### Overview

![infogan architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Noise + Latent Code | `input` | shape=[110] |
| 2 | Generator | `conv2d` | outFeatures=3 |
| 3 | Discriminator | `conv2d` | outFeatures=1 |
| 4 | Q Network (code recovery) | `linear` | outFeatures=10 |
| 5 | Recovered Code | `output` | — |

</details>

### Components

1. **Noise + Latent Code** — input layer. 2. **Generator** — conv2d layer. 2. **Discriminator** — conv2d layer. 2. **Q Network (code recovery)** — linear layer. 2. **Recovered Code** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — GAN with disentangled latent codes: maximizes mutual information between latent codes and generated data.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
