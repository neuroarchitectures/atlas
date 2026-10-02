# Architecture: Cascade R-CNN

## Motivation

Multi-stage detector with cascaded bounding box heads at increasing IoU thresholds for high-quality detection.

## Core Idea

Multi-stage detector with cascaded bounding box heads at increasing IoU thresholds for high-quality detection.

## Architecture

### Overview

![cascade-rcnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 800, 800] |
| 2 | ResNet Backbone | `conv2d` | outFeatures=2048 |
| 3 | Feature Pyramid Network | `conv2d` | outFeatures=256 |
| 4 | Region Proposal Network | `conv2d` | outFeatures=256 |
| 5 | RoI Align | `roialign` | outFeatures=256 |
| 6 | BBox Head Stage 1 (IoU 0.5) | `linear` | outFeatures=1024 |
| 7 | BBox Head Stage 2 (IoU 0.6) | `linear` | outFeatures=1024 |
| 8 | BBox Head Stage 3 (IoU 0.7) | `linear` | outFeatures=81 |
| 9 | Detections | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **ResNet Backbone** — conv2d layer. 2. **Feature Pyramid Network** — conv2d layer. 2. **Region Proposal Network** — conv2d layer. 2. **RoI Align** — roialign layer. 2. **BBox Head Stage 1 (IoU 0.5)** — linear layer. 2. **BBox Head Stage 2 (IoU 0.6)** — linear layer. 2. **BBox Head Stage 3 (IoU 0.7)** — linear layer. 2. **Detections** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Multi-stage detector with cascaded bounding box heads at increasing IoU thresholds for high-quality detection.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
