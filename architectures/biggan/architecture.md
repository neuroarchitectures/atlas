# Architecture: BigGAN

## Motivation

GANs had not been shown to benefit from large-scale training like classification models. BigGAN demonstrates that scaling GANs (larger batch, wider channels, self-attention) dramatically improves sample quality.

## Core Idea

Use a class-conditional BigGAN generator with self-attention blocks, hierarchical latent injection (noise split and injected at multiple layers), and spectral normalization for training stability at scale.

## Architecture

### Overview

![biggan architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent z | `input` | shape: [128] |
| 2 | Class Embedding | `embedding` | numEmbeddings: 1000, embeddingDim: 128 |
| 3 | Split z | `identity` | hierarchical noise injection |
| 4 | BigGAN Block 1 | `convTranspose` | class-conditional residual block |
| 5 | Self-Attention | `attention` | non-local self-attention block |
| 6 | BigGAN Block 2 | `convTranspose` | class-conditional residual block |
| 7 | Generated Image | `output` | toRGB via 1x1 conv |

</details>

### Components

1. **Hierarchical noise** — The latent z is split into chunks, each injected into a different generator block via class-conditional batch norm. 2. **Self-attention** — Non-local self-attention block at intermediate resolution for long-range dependency modeling. 3. **Spectral normalization** — Applied to all layers in both G and D for training stability. 4. **Class-conditional BN** — Conditional batch norm with class embedding and noise chunk for scale/bias. 5. **Truncation trick** — At inference, sample z from a truncated normal for quality/diversity tradeoff.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a class-conditional BigGAN generator with self-attention blocks, hierarchical latent injection (noise split and injected at multiple layers), and spectral normalization for training stability at s
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SAGAN (self-attention GAN). Successor: BigGAN-Deep, GANs at scale.

## References

- Brock et al. 2019
