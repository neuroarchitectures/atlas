# Depthwise-Separable Convolution

## Design Philosophy

Split a standard convolution into two cheaper steps: a **depthwise conv** (one filter per input channel — spatial mixing only, no cross-channel mixing) followed by a **pointwise 1×1 conv** (channel mixing only). The philosophy is **factorization**: a k×k×C_in×C_out conv costs `k²·C_in·C_out` per location; the factorized version costs `k²·C_in + C_in·C_out`, a reduction of roughly `1/C_out + 1/k²`. For 3×3 with 256 channels, that's ~8–9× fewer operations.

## Functionality

1. **Depthwise conv2d**: kernel `[k, k]` with `groups = inChannels` — each input channel gets its own k×k filter. Output channels = input channels × depthMultiplier.
2. **Pointwise conv2d**: 1×1 conv that mixes across channels, projecting to the desired output channel count.
3. Optional BN + activation between/after.

## Used By

| Model | Role |
|-------|------|
| MobileNetV2 | The cheap spatial conv inside every inverted-residual block |
| EfficientNet-B0 | Inside MBConv blocks |
| EEGNet | Temporal + spatial depthwise convs (spatial filter across electrodes) |
| Xception | Replaces Inception modules with depthwise-separable |
| ConvNeXt | 7×7 depthwise conv (large kernel) + 1×1 pointwise |

## Features

- **Efficiency**: 8–9× fewer multiply-adds than standard conv for typical 3×3 settings.
- **Separation of concerns**: Spatial mixing (depthwise) and channel mixing (pointwise) are decoupled, each learnable independently.
- **Mobile-friendly**: The foundation of on-device CNNs.

## Evolution

- **Predecessor**: Standard convolution (does both spatial + channel mixing jointly).
- **Successor**: Inverted residual block (MobileNetV2) wraps depthwise-separable in an expand-project structure; ConvNeXt scales the depthwise kernel to 7×7.
