# Architecture: SSD

## Motivation

Single Shot MultiBox Detector: multi-scale single-stage detector with default boxes and feature pyramid.

## Core Idea

Single Shot MultiBox Detector: multi-scale single-stage detector with default boxes and feature pyramid.

## Architecture

### Overview

![ssd architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 300, 300] |
| 2 | VGG Backbone | `conv2d` | outFeatures=512 |
| 3 | Feature Map 1 (38x38) | `conv2d` | outFeatures=1024 |
| 4 | Feature Map 2 (19x19) | `conv2d` | outFeatures=512 |
| 5 | Feature Map 3 (10x10) | `conv2d` | outFeatures=256 |
| 6 | Feature Map 4 (5x5) | `conv2d` | outFeatures=256 |
| 7 | Feature Map 5 (3x3) | `conv2d` | outFeatures=256 |
| 8 | Feature Map 6 (1x1) | `conv2d` | outFeatures=256 |
| 9 | Multi-scale Detection Head | `conv2d` | outFeatures=84 |
| 10 | Detections | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **VGG Backbone** — conv2d layer. 2. **Feature Map 1 (38x38)** — conv2d layer. 2. **Feature Map 2 (19x19)** — conv2d layer. 2. **Feature Map 3 (10x10)** — conv2d layer. 2. **Feature Map 4 (5x5)** — conv2d layer. 2. **Feature Map 5 (3x3)** — conv2d layer. 2. **Feature Map 6 (1x1)** — conv2d layer. 2. **Multi-scale Detection Head** — conv2d layer. 2. **Detections** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Single Shot MultiBox Detector: multi-scale single-stage detector with default boxes and feature pyramid.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
