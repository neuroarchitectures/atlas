# Architecture: TransUNet

## Motivation

U-Net with ViT encoder for medical image segmentation: combines CNN local features with transformer global context.

## Core Idea

U-Net with ViT encoder for medical image segmentation: combines CNN local features with transformer global context.

## Architecture

### Overview

![transunet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 224, 224] |
| 2 | CNN Encoder | `conv2d` | outFeatures=512 |
| 3 | ViT Encoder | `transformer-encoder` | numLayers=12, hiddenSize=768, numHeads=12 |
| 4 | U-Net Decoder | `conv2d` | outFeatures=64 |
| 5 | Segmentation Head | `conv2d` | outFeatures=4 |
| 6 | Segmentation Map | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **CNN Encoder** — conv2d layer. 2. **ViT Encoder** — transformer-encoder layer. 2. **U-Net Decoder** — conv2d layer. 2. **Segmentation Head** — conv2d layer. 2. **Segmentation Map** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — U-Net with ViT encoder for medical image segmentation: combines CNN local features with transformer global context.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
