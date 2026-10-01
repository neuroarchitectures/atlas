# Architecture: SENet

## Motivation

CNNs treat all channels equally, but different channels carry different importance for the current input. SENet introduces a lightweight channel attention mechanism.

## Core Idea

Squeeze (global average pooling to get per-channel descriptor) then Excitation (two FC layers with ReLU + sigmoid to produce per-channel weights). Multiply original features by these weights.

## Architecture

### Overview

![senet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Feature Map | `input` | shape: [256, 14, 14] |
| 2 | Conv | `conv` | standard conv |
| 3 | Squeeze (GAP) | `identity` | global average pooling -> [256] |
| 4 | Excite (FC1) | `linear` | outFeatures: 16 (reduction ratio 16:1) |
| 5 | ReLU | `relu` |  |
| 6 | Excite (FC2) | `linear` | outFeatures: 256 |
| 7 | Sigmoid | `sigmoid` | per-channel weights [0,1] |
| 8 | Scale | `multiply` | channel-wise multiplication |
| 9 | Recalibrated | `output` |  |

</details>

### Components

1. **Squeeze** — Global average pooling over spatial dimensions, producing a per-channel descriptor vector. 2. **Excitation** — Two FC layers with a bottleneck (reduction ratio r=16), ReLU, and sigmoid. Produces per-channel attention weights. 3. **Scale** — Element-wise multiplication of the original feature map by the channel weights. 4. **Integration** — SE blocks can be inserted into any CNN (ResNet, Inception, etc.) with minimal overhead.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Squeeze (global average pooling to get per-channel descriptor) then Excitation (two FC layers with ReLU + sigmoid to produce per-channel weights). Multiply original features by these weights.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ResNet. Successor: SKNet, CBAM, Coordinate Attention.

## References

- Hu et al. 2018
