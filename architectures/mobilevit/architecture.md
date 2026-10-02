# Architecture: MobileViT

## Motivation

Mobile CNNs are efficient but lack global context. ViTs have global context but are too expensive for mobile.

## Core Idea

Replace local processing in CNNs with Transformer-based global processing in MobileViT blocks, combining CNN efficiency with Transformer global representation.

## Architecture

### Overview

![mobilevit architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 256, 256] |
| 2 | Conv 3x3 | `conv2d` | stride: 2 |
| 3 | MV2 Block | `mv2-block` |  |
| 4 | MobileViT Block | `mobilevit-block` |  |
| 5 | MV2 Block | `mv2-block` | stride: 2 |
| 6 | MobileViT Block | `mobilevit-block` |  |
| 7 | MV2 Block | `mv2-block` | stride: 2 |
| 8 | MobileViT Block | `mobilevit-block` |  |
| 9 | Conv 1x1 | `conv2d` |  |
| 10 | Global Avg Pool | `pooling` |  |
| 11 | Classifier | `linear` | outFeatures: 1000 |
| 12 | Class Logits | `output` |  |

</details>

### Components

1. **MobileNetV2 blocks** — Inverted residual with depthwise separable convolutions. 2. **MobileViT blocks** — Unfold patches into sequences, apply transformer, fold back. 3. **Hybrid structure** — CNN for early stages, MobileViT for later stages. 4. **Classifier** — Conv 1x1 + global avg pool + linear.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace local processing in CNNs with Transformer-based global processing in MobileViT blocks, combining CNN efficiency 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MobileNetV2, ViT. Successor: MobileViT-V2.

## References

Mehta & Rastegari 2021
