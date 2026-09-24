# Bottleneck Residual Block

## Design Philosophy

The 1×1 → 3×3 → 1×1 sandwich that made 50+ layer ResNets practical. The philosophy is **bottleneck design**: reduce channels to a narrow 3×3 conv (cheap), then expand back. Two 1×1 convs are far cheaper than two 3×3 convs at full width, so the block achieves the same receptive field at a fraction of the FLOPs. The identity skip still connects the wide input to the wide output.

## Functionality

Structure: `input → 1×1 conv-BN-ReLU → 3×3 conv-BN-ReLU → 1×1 conv-BN → add(input) → ReLU → output`.

- **1×1 reduce**: Projects from `C` channels down to `C/4` (the bottleneck).
- **3×3 conv**: Spatial mixing at the narrow bottleneck width.
- **1×1 expand**: Projects back up to `4C` (or to the output channel count).
- The skip is identity when shapes match, or a 1×1 projection when they change.

## Used By

| Model | Role |
|-------|------|
| ResNet-50 / 101 / 152 | The standard block, stacked 16/23/50 times |
| Mask R-CNN | Backbone (ResNet-FPN uses these) |
| Faster R-CNN | Backbone feature extractor |

## Features

- **Parameter efficiency**: ~4× fewer FLOPs than two 3×3 convs at full width.
- **Deeper networks**: The narrow middle enables stacking 50–152 layers without exploding cost.
- **Pre-activation variant**: ResNet-v2 moves BN-ReLU *before* the conv (pre-norm), improving gradient flow.

## Evolution

- **Predecessor**: The basic residual block (two 3×3 convs).
- **Successor**: Inverted residual (MobileNetV2) — flips the bottleneck (narrow→wide→narrow), connecting the narrow ends.
