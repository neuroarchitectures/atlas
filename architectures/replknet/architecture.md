# Architecture: RepLKNet

## Motivation

CNNs use small kernels (3x3, 7x7) for efficiency, but ViTs achieve global context through attention. Large kernels could bridge this gap but are computationally expensive.

## Core Idea

Use structural re-parameterization to decompose large kernels into parallel small kernels during training, then merge them at inference for efficient large-kernel convolution.

## Architecture

### Overview

![replknet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Stem | `conv2d` | kernelSize: 13 |
| 3 | Stage 1 | `replk-block` | kernel: 13 |
| 4 | Downsample | `conv2d` |  |
| 5 | Stage 2 | `replk-block` | kernel: 13 |
| 6 | Downsample | `conv2d` |  |
| 7 | Stage 3 | `replk-block` | kernel: 13 |
| 8 | Downsample | `conv2d` |  |
| 9 | Stage 4 | `replk-block` | kernel: 13 |
| 10 | Global Avg Pool | `pooling` |  |
| 11 | Classifier | `linear` | outFeatures: 1000 |
| 12 | Class Logits | `output` |  |

</details>

### Components

1. **Large kernel blocks** — Parallel small conv + identity + large conv, merged at inference. 2. **Structural re-parameterization** — Train with decomposed kernels, merge at inference. 3. **Four stages** — Hierarchical downsampling. 4. **Classifier** — Global avg pool + linear.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use structural re-parameterization to decompose large kernels into parallel small kernels during training, then merge th
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ConvNeXt. Successor: InternImage.

## References

Ding et al. 2022
