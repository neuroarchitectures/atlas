# Architecture: FCOS

## Motivation

Fully Convolutional One-Stage Detector: anchor-free per-pixel prediction with center-ness branch.

## Core Idea

Fully Convolutional One-Stage Detector: anchor-free per-pixel prediction with center-ness branch.

## Architecture

### Overview

![fcos architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 800, 800] |
| 2 | ResNet Backbone | `conv2d` | outFeatures=2048 |
| 3 | Feature Pyramid Network | `conv2d` | outFeatures=256 |
| 4 | Classification Head | `conv2d` | outFeatures=80 |
| 5 | Regression Head | `conv2d` | outFeatures=4 |
| 6 | Center-ness Head | `conv2d` | outFeatures=1 |
| 7 | Detections | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **ResNet Backbone** — conv2d layer. 2. **Feature Pyramid Network** — conv2d layer. 2. **Classification Head** — conv2d layer. 2. **Regression Head** — conv2d layer. 2. **Center-ness Head** — conv2d layer. 2. **Detections** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Fully Convolutional One-Stage Detector: anchor-free per-pixel prediction with center-ness branch.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
