# Inverted Residual Block (MobileNetV2)

## Design Philosophy

Flip the ResNet bottleneck: instead of wide→narrow→wide, go **narrow→wide→narrow**, and connect the skip between the narrow ends. The philosophy: in low-dimensional space, ReLU destroys information (linear bottleneck theorem), so do the expensive spatial work in a *high*-dimensional expanded space, and keep the skip connection in the low-dimensional manifold where the information actually lives.

## Functionality

Structure: `input(narrow) → 1×1 expand → BN-ReLU → depthwise 3×3 → BN-ReLU → 1×1 project(linear) → output(narrow) → add(input)`.

1. **1×1 expand**: Project from `C` channels up to `kC` (e.g., 6× expansion).
2. **Depthwise 3×3**: Spatial filtering at the wide, high-dimensional representation (where ReLU is safe).
3. **1×1 project**: Compress back down to `C` channels — **no activation** (linear bottleneck), preserving information in low-dim space.
4. Skip connection between the narrow input and narrow output.

## Used By

| Model | Role |
|-------|------|
| MobileNetV2 | 17 stacked inverted-residual blocks (3.5M params) |
| EfficientNet-B0 | MBConv blocks (inverted residual + squeeze-excite) |
| MobileNetV3 | Same block with hardware-aware search + SE |

## Features

- **Linear bottleneck**: No ReLU after the final projection — the key insight that prevents information loss in low dimensions.
- **Expand-then-compress**: The expensive depthwise conv runs in a wide space; the residual lives in a narrow space.
- **Parameter economy**: 3.5M params for ImageNet-grade accuracy, deployable on mobile.

## Evolution

- **Predecessor**: MobileNetV1 (depthwise-separable, no residual) and ResNet bottleneck (wide→narrow→wide).
- **Successor**: EfficientNet's MBConv adds a squeeze-and-excite gate inside each block; MobileNetV3 uses NAS to tune it.
