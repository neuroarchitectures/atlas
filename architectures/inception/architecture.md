# Architecture: Inception (GoogLeNet)

## Motivation

Improving vision model quality typically means increasing depth and width, but naively scaling up dense layers leads to huge parameter counts and computational cost. Inception addresses this by computing multi-scale features in parallel within each module, keeping computation sparse.

## Core Idea

The Inception module computes 1x1, 3x3, 5x5 convolutions and max pooling in parallel, then concatenates results along the channel dimension. 1x1 convolutions are used as dimensionality reduction before expensive 3x3/5x5 convolutions, drastically reducing computation.

## Architecture

### Overview

![inception architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (11 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 3, 224, 224] |
| 2 | 1x1 conv | `conv2d` | outChannels: 64, kernelSize: 1, stride: 1, inChannels: 3 |
| 3 | 1x1 conv | `conv2d` | outChannels: 96, kernelSize: 1, stride: 1, inChannels: 3 |
| 4 | 1x1 conv | `conv2d` | outChannels: 16, kernelSize: 1, stride: 1, inChannels: 3 |
| 5 | 3x3 maxpool | `maxpool2d` | kernelSize: 3, stride: 1, padding: 1 |
| 6 | 3x3 conv | `conv2d` | outChannels: 128, kernelSize: 3, stride: 1, padding: 1, inChannels: 96 |
| 7 | 5x5 conv | `conv2d` | outChannels: 32, kernelSize: 5, stride: 1, padding: 2, inChannels: 16 |
| 8 | 1x1 conv | `conv2d` | outChannels: 64, kernelSize: 1, stride: 1, inChannels: 3 |
| 9 | Concatenate | `custom` | type: channel_concat |
| 10 | ReLU | `relu` |  |
| 11 | Output | `output` |  |

</details>

The Inception module computes 1x1, 3x3, 5x5 convolutions and max pooling in parallel, then concatenates results along the channel dimension. 1x1 convolutions are used as dimensionality reduction before expensive 3x3/5x5 convolutions, drastically reducing computation.

### Components

2. **1x1 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 64, kernelSize: 1, stride: 1, inChannels: 3
3. **1x1 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 96, kernelSize: 1, stride: 1, inChannels: 3
4. **1x1 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 16, kernelSize: 1, stride: 1, inChannels: 3
5. **3x3 maxpool** (`maxpool2d`, scope: `inception`) — Params: kernelSize: 3, stride: 1, padding: 1
6. **3x3 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 128, kernelSize: 3, stride: 1, padding: 1, inChannels: 96
7. **5x5 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 32, kernelSize: 5, stride: 1, padding: 2, inChannels: 16
8. **1x1 conv** (`conv2d`, scope: `inception`) — Params: outChannels: 64, kernelSize: 1, stride: 1, inChannels: 3
9. **Concatenate** (`custom`, scope: `inception`) — Params: type: channel_concat
10. **ReLU** (`relu`, scope: `inception`) — Params: none

### Data Flow

The architecture processes input through a sequence of 11 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Inception (GoogLeNet).

## Evolution

Inception (GoogLeNet) introduced multi-scale parallel processing. Successors: Inception v2/v3 (factorized convolutions), Inception-v4, Inception-ResNet, Xception (extreme depthwise separable). The dimensionality-reduction via 1x1 convolutions influenced MobileNet and EfficientNet.

## Source

- **Paper:** arXiv:1409.4842
- **Year:** 2015
- **Authors:** Szegedy et al.
