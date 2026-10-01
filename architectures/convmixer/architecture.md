# Architecture: ConvMixer

## Motivation

MLP-Mixer showed pure MLPs can work for vision. ConvMixer asks: can pure convolutions (no attention, no MLP mixing) achieve similar results with better efficiency?

## Core Idea

Patch embeddings via strided convolution. Then alternate between depthwise convolutions (spatial mixing within each channel) and pointwise convolutions (channel mixing). No attention or MLP blocks.

## Architecture

### Overview

![convmixer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Patch Embed | `conv` | kernelSize: 7, stride: 7, 256 channels |
| 3 | ConvMixer Block | `conv` | depthwise 7x7 + pointwise 1x1, repeated |
| 4 | Output | `output` |  |

</details>

### Components

1. **Depthwise convolution** — 7x7 depthwise conv mixes spatial information within each channel independently. 2. **Pointwise convolution** — 1x1 pointwise conv mixes channels. 3. **Residual** — Depthwise conv has a residual connection. 4. **GELU activation** — Used after both depthwise and pointwise convolutions. 5. **BatchNorm** — Used instead of LayerNorm.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Patch embeddings via strided convolution. Then alternate between depthwise convolutions (spatial mixing within each channel) and pointwise convolutions (channel mixing). No attention or MLP blocks.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MLP-Mixer, ViT. Successor: ConvMixer variants, CoAtNet.

## References

- Trockman & Kolter 2022
