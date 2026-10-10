# Architecture: UNet++

## Motivation

Nested U-Net with dense skip connections and deep supervision for improved segmentation.

## Core Idea

Nested U-Net with dense skip connections and deep supervision for improved segmentation.

## Architecture

### Overview

![unet-plus-plus architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 256, 256] |
| 2 | Encoder Block 1 | `conv2d` | outFeatures=64 |
| 3 | Encoder Block 2 | `conv2d` | outFeatures=128 |
| 4 | Encoder Block 3 | `conv2d` | outFeatures=256 |
| 5 | Encoder Block 4 | `conv2d` | outFeatures=512 |
| 6 | Bottleneck | `conv2d` | outFeatures=512 |
| 7 | Nested Decoder 3-1 | `conv2d` | outFeatures=256 |
| 8 | Nested Decoder 2-1 | `conv2d` | outFeatures=128 |
| 9 | Nested Decoder 1-1 | `conv2d` | outFeatures=64 |
| 10 | Segmentation Map | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Encoder Block 1** — conv2d layer. 2. **Encoder Block 2** — conv2d layer. 2. **Encoder Block 3** — conv2d layer. 2. **Encoder Block 4** — conv2d layer. 2. **Bottleneck** — conv2d layer. 2. **Nested Decoder 3-1** — conv2d layer. 2. **Nested Decoder 2-1** — conv2d layer. 2. **Nested Decoder 1-1** — conv2d layer. 2. **Segmentation Map** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Nested U-Net with dense skip connections and deep supervision for improved segmentation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
