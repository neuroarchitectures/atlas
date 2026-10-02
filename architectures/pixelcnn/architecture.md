# Architecture: PixelCNN

## Motivation

Autoregressive pixel-by-pixel image generation with masked convolutions.

## Core Idea

Autoregressive pixel-by-pixel image generation with masked convolutions.

## Architecture

### Overview

![pixelcnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 64, 64] |
| 2 | Masked Conv A | `conv2d` | outFeatures=128 |
| 3 | Masked Conv B x12 | `conv2d` | outFeatures=128 |
| 4 | Output Head | `conv2d` | outFeatures=256 |
| 5 | Pixel Probabilities | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Masked Conv A** — conv2d layer. 2. **Masked Conv B x12** — conv2d layer. 2. **Output Head** — conv2d layer. 2. **Pixel Probabilities** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Autoregressive pixel-by-pixel image generation with masked convolutions.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
