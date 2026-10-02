# Architecture: Sparse R-CNN

## Motivation

Traditional detectors (Faster R-CNN, Mask R-CNN) use dense proposals and heavy ROI heads. Sparse R-CNN uses a fixed set of learnable proposal features (e.g., 100) with dynamic heads, simplifying the pipeline.

## Core Idea

A set of N learnable proposal embeddings (N=100-300) are initialized as parameters. Each proposal interacts with image features through dynamic heads (generated from the proposal itself). No RPN, no NMS needed.

## Architecture

### Overview

![sparse-rcnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3xHxW |
| 2 | ResNet + FPN | `conv2d` | multi-scale features |
| 3 | Dynamic Instance Heads | `linear` | 100 proposals, 6 stages |
| 4 | Detection | `output` | 100 boxes + classes |

</details>

### Components

1. **Learnable proposals** — N=100 proposal embeddings as learnable parameters. 2. **Proposal boxes** — initial box estimates as learnable parameters. 3. **Dynamic heads** — head weights generated from proposal features. 4. **Self-attention** — proposals interact via self-attention. 5. **No NMS** — set prediction, no post-processing. 6. **End-to-end** — fully differentiable.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A set of N learnable proposal embeddings (N=100-300) are initialized as parameters. Each proposal interacts with image f...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DETR (set prediction), Faster R-CNN. Successor: Cascade Sparse R-CNN.

## References

- Sun et al. 2021
