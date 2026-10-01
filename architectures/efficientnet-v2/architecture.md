# Architecture: EfficientNetV2

## Motivation

EfficientNet (V1) has slow training due to depthwise convolutions in early layers and fixed image size. EfficientNetV2 uses NAS to find a better architecture with fused MBConv and progressive training.

## Core Idea

Replace early MBConv with Fused-MBConv (3x3 conv + 1x1 conv instead of 1x1 + 3x3 depthwise + 1x1). Add progressive learning (increase image size and regularization during training).

## Architecture

### Overview

![efficientnetv2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Stem Conv | `conv` | 3x3, stride 2 |
| 3 | Fused-MBConv 1 | `conv` | early layers: 3x3 + 1x1 (no depthwise) |
| 4 | MBConv 1 | `conv` | later layers: 1x1 + DW 3x3 + 1x1 |
| 5 | Output | `output` |  |

</details>

### Components

1. **Fused-MBConv** — Replaces the 1x1 expand + 3x3 depthwise + 1x1 project with a single 3x3 conv + 1x1 conv. Faster in early layers where channels are small. 2. **MBConv** — Standard inverted bottleneck with depthwise conv, used in later stages. 3. **Progressive learning** — Start with small images and weak regularization, gradually increase both. 4. **NAS** — Architecture found via neural architecture search with training speed as an objective.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace early MBConv with Fused-MBConv (3x3 conv + 1x1 conv instead of 1x1 + 3x3 depthwise + 1x1). Add progressive learning (increase image size and regularization during training).
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: EfficientNet. Successor: EfficientNetV2 variants.

## References

- Tan & Le 2021
