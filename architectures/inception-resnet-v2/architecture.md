# Architecture: Inception-ResNet V2

## Motivation

Inception networks achieved excellent performance but training deep Inception networks is difficult. Residual connections could help.

## Core Idea

Combine Inception modules with residual connections, enabling deeper and more efficient training while maintaining the multi-scale processing of Inception.

## Architecture

### Overview

![inception-resnet-v2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 299, 299] |
| 2 | Stem | `conv2d` |  |
| 3 | Stem | `conv2d` |  |
| 4 | Stem | `conv2d` |  |
| 5 | Inception-ResNet A | `inception-resnet` | 5x |
| 6 | Reduction A | `conv2d` |  |
| 7 | Inception-ResNet B | `inception-resnet` | 10x |
| 8 | Reduction B | `conv2d` |  |
| 9 | Inception-ResNet C | `inception-resnet` | 5x |
| 10 | Global Avg Pool | `pooling` |  |
| 11 | Classifier | `linear` | outFeatures: 1000 |
| 12 | Class Logits | `output` |  |

</details>

### Components

1. **Stem** — Initial convolutional layers for feature extraction. 2. **Inception-ResNet blocks** — Inception modules with residual connections (A, B, C types). 3. **Reduction blocks** — Downsample spatial dimensions. 4. **Global avg pool** — Spatial aggregation. 5. **Classifier** — Linear layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Combine Inception modules with residual connections, enabling deeper and more efficient training while maintaining the m
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Inception-V3, ResNet. Successor: Inception-V4.

## References

Szegedy et al. 2017
