# MBConv Block

## Design Philosophy

MobileNetV2's inverted-residual block with a **squeeze-and-excite (SE) channel-attention gate** inserted after the depthwise conv. The philosophy: the inverted residual gives you efficiency; SE gives you dynamic, input-dependent channel reweighting for free (a tiny MLP). NAS (EfficientNet) discovered that this combination, compound-scaled across depth/width/resolution, is the accuracy-per-FLOP sweet spot.

## Functionality

Structure: `input → 1×1 expand → BN-Swish → depthwise k×k → BN-Swish → SE → 1×1 project(linear) → add(input)`.

1. **1×1 expand** (ratio 6×): widen the representation.
2. **Depthwise k×k**: spatial filtering in the wide space.
3. **Squeeze-and-Excite**: global avg pool → FC-ReLU-FC-sigmoid → channel-wise scale. Recalibrates channels based on global context.
4. **1×1 project**: compress back (linear).
5. Skip when input/output shapes match.

## Used By

| Model | Role |
|-------|------|
| EfficientNet-B0–B7 | The building block, compound-scaled |
| EfficientNetV2 | Fused-MBConv variant (early layers fuse expand+depthwise) |
| MobileNetV3 | SE-augmented inverted residual |

## Features

- **SE attention**: A ~1% parameter overhead SE block that consistently improves accuracy.
- **Swish activation**: `x·sigmoid(x)` instead of ReLU, smoother gradients.
- **Compound scaling**: A single coefficient φ scales depth, width, and resolution together, yielding B0→B7.

## Evolution

- **Predecessor**: Inverted residual (MobileNetV2) + standalone SE-Net.
- **Successor**: EfficientNetV2 replaces early MBConv with fused-MBConv (faster on TPUs) and uses progressive learning.
