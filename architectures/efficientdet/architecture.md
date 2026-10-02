# Architecture: EfficientDet

## Motivation

Scalable object detector with BiFPN multi-scale feature fusion and compound scaling.

## Core Idea

Scalable object detector with BiFPN multi-scale feature fusion and compound scaling.

## Architecture

### Overview

![efficientdet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 512, 512] |
| 2 | EfficientNet Backbone | `conv2d` | outFeatures=320 |
| 3 | BiFPN Feature Fusion | `conv2d` | outFeatures=64 |
| 4 | Class Head | `conv2d` | outFeatures=90 |
| 5 | Box Head | `conv2d` | outFeatures=4 |
| 6 | Detections | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **EfficientNet Backbone** — conv2d layer. 2. **BiFPN Feature Fusion** — conv2d layer. 2. **Class Head** — conv2d layer. 2. **Box Head** — conv2d layer. 2. **Detections** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Scalable object detector with BiFPN multi-scale feature fusion and compound scaling.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
