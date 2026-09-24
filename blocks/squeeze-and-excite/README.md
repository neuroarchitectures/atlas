# Squeeze-and-Excite (SE) Block

## Design Philosophy

Explicitly model **channel-wise dependencies** with a gating mechanism. A standard conv treats all channels equally; SE learns to recalibrate them per-input. The "squeeze" (global avg pool) collapses spatial dimensions into a per-channel descriptor; the "excite" (a bottleneck FC → ReLU → FC → sigmoid) produces per-channel weights that scale the original feature map. The philosophy: *which channels matter is input-dependent*.

## Functionality

Structure: `input → globalAvgPool → FC(C → C/r) → ReLU → FC(C/r → C) → sigmoid → channelScale(input)`.

1. **Squeeze**: Global average pooling over spatial dims → `[1, 1, C]` descriptor.
2. **Excite**: Two FC layers with a bottleneck ratio `r` (e.g., 16) and a sigmoid, producing weights in [0,1] per channel.
3. **Scale**: Multiply the original feature map by these per-channel weights.

## Used By

| Model | Role |
|-------|------|
| SE-Net (SE-ResNet) | SE-Residual blocks — won ImageNet 2017 |
| EfficientNet / MBConv | Inside every MBConv block |
| MobileNetV3 | SE inside inverted residuals |

## Features

- **Cheap**: Parameter overhead ~`2C²/r` — negligible vs. the conv stack.
- **Universal**: Can be dropped into almost any conv-based architecture.
- **Performance**: Consistent ~1% ImageNet top-1 gain for minimal cost.

## Evolution

- **Predecessor**: Attention mechanisms in NLP; channel attention in STN.
- **Successor**: CBAM (adds spatial attention); and the broader attention-block family. Conceptually a precursor to the attention mechanism in Transformers (attention over channels, not positions).
