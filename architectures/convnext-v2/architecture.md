# Architecture: ConvNeXt V2

## Motivation

ConvNeXt closed the gap with ViTs, but self-supervised pre-training (like MAE) was not effective for ConvNets.

## Core Idea

Introduce FCMAE (Fully Convolutional Masked Autoencoder) for ConvNet self-supervised pre-training, plus Global Response Normalization (GRN) to enhance channel competition.

## Architecture

### Overview

![convnext-v2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (13 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Stem | `conv2d` | stride: 4 |
| 3 | Stage 1 | `convnext-block` | depth: 3 |
| 4 | Downsample | `conv2d` |  |
| 5 | Stage 2 | `convnext-block` | depth: 3 |
| 6 | Downsample | `conv2d` |  |
| 7 | Stage 3 | `convnext-block` | depth: 9 |
| 8 | Downsample | `conv2d` |  |
| 9 | Stage 4 | `convnext-block` | depth: 3 |
| 10 | GRN | `grn` |  |
| 11 | Global Avg Pool | `pooling` |  |
| 12 | Classifier | `linear` | outFeatures: 1000 |
| 13 | Class Logits | `output` |  |

</details>

### Components

1. **ConvNeXt backbone** — 4-stage hierarchical ConvNet with depthwise separable convolutions. 2. **GRN** — Global response normalization adds channel competition. 3. **FCMAE** — Masked autoencoding with convolutions for self-supervised pre-training. 4. **Classifier head** — Global avg pool + linear.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Introduce FCMAE (Fully Convolutional Masked Autoencoder) for ConvNet self-supervised pre-training, plus Global Response 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ConvNeXt, MAE. Successor: ConvNeXt-V3.

## References

Woo et al. 2023
