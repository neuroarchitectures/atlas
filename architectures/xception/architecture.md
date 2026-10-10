# Architecture: Xception

## Motivation

The Inception module essentially performs depthwise separable convolution in a multi-branch form. Xception pushes this to the extreme by fully decoupling spatial and channel-wise learning, using depthwise separable convolutions as the core building block.

## Core Idea

Replace Inception modules with depthwise separable convolutions: (1) depthwise convolution (one filter per channel for spatial learning), followed by (2) pointwise 1x1 convolution (for channel mixing). This extreme separation of spatial and channel computation improves efficiency and accuracy.

## Architecture

### Overview

![xception architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (11 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 3, 299, 299] |
| 2 | Entry 3x3 conv | `conv2d` | outChannels: 32, kernelSize: 3, stride: 2, inChannels: 3 |
| 3 | Entry 3x3 conv | `conv2d` | outChannels: 64, kernelSize: 3, stride: 1, inChannels: 32 |
| 4 | SeparableConv Block | `custom` | type: separable_conv, outChannels: 128 |
| 5 | SeparableConv Block | `custom` | type: separable_conv, outChannels: 256 |
| 6 | SeparableConv Block | `custom` | type: separable_conv, outChannels: 728 |
| 7 | Middle Flow (x8) | `custom` | type: separable_conv_x8, outChannels: 728 |
| 8 | Exit SeparableConv | `custom` | type: separable_conv, outChannels: 1024 |
| 9 | Global AvgPool | `avgpool2d` | type: global |
| 10 | FC | `linear` | outFeatures: 1000, inFeatures: 1024 |
| 11 | Output | `output` |  |

</details>

Replace Inception modules with depthwise separable convolutions: (1) depthwise convolution (one filter per channel for spatial learning), followed by (2) pointwise 1x1 convolution (for channel mixing). This extreme separation of spatial and channel computation improves efficiency and accuracy.

### Components

2. **Entry 3x3 conv** (`conv2d`, scope: `entry`) — Params: outChannels: 32, kernelSize: 3, stride: 2, inChannels: 3
3. **Entry 3x3 conv** (`conv2d`, scope: `entry`) — Params: outChannels: 64, kernelSize: 3, stride: 1, inChannels: 32
4. **SeparableConv Block** (`custom`, scope: `block.0`) — Params: type: separable_conv, outChannels: 128
5. **SeparableConv Block** (`custom`, scope: `block.1`) — Params: type: separable_conv, outChannels: 256
6. **SeparableConv Block** (`custom`, scope: `block.2`) — Params: type: separable_conv, outChannels: 728
7. **Middle Flow (x8)** (`custom`, scope: `middle`) — Params: type: separable_conv_x8, outChannels: 728
8. **Exit SeparableConv** (`custom`, scope: `exit`) — Params: type: separable_conv, outChannels: 1024
9. **Global AvgPool** (`avgpool2d`, scope: `exit`) — Params: type: global
10. **FC** (`linear`, scope: `classifier`) — Params: outFeatures: 1000, inFeatures: 1024

### Data Flow

The architecture processes input through a sequence of 11 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Xception.

## Evolution

Xception is an extreme evolution of Inception, fully decoupling spatial and channel computation. It directly inspired MobileNetV2/V3 (inverted residuals + depthwise separable convs) and EfficientNet. The depthwise separable convolution became a standard building block for efficient vision architectures.

## Source

- **Paper:** arXiv:1610.02357
- **Year:** 2017
- **Authors:** Chollet
