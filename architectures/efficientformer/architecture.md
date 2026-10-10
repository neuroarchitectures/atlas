# Architecture: EfficientFormer

## Motivation

ViTs are fast on GPUs but slow on mobile devices due to token-mixing operations. EfficientFormer designs a dimension-consistent transformer where operations can be converted to tensor reshaping, enabling mobile-efficient inference.

## Core Idea

Uses a dimension-consistent design: 3D tensors (C×H×W) for conv-like operations and 2D (N×C) for attention. Meta-former architecture with pooling or attention as token mixer. Some blocks run in 2D (fast), others in 3D (conv-compatible). Latency-driven search finds optimal architecture.

## Architecture

### Overview

![efficientformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x224x224 |
| 2 | Patch Embed | `identity` | 4x4 patches |
| 3 | EfficientFormer Blocks | `linear` | 4 stages, mixed 2D/3D |
| 4 | Classification | `output` | 1000 classes |

</details>

### Components

1. **Dimension-consistent** — 3D (C×H×W) and 2D (N×C) forms. 2. **Meta-former** — pooling or attention as token mixer. 3. **Mobile-efficient** — avoids reshape overhead on mobile. 4. **Latency-driven search** — search finds mobile-optimal architecture. 5. **Hybrid blocks** — some 2D (fast), some 3D (conv-compatible).

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Uses a dimension-consistent design: 3D tensors (C×H×W) for conv-like operations and 2D (N×C) for attention. Meta-former ...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MobileViT, PoolFormer. Successor: EfficientFormer v2.

## References

- Li et al. 2022
