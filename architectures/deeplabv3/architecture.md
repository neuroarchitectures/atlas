# Architecture: DeepLab v3

## Motivation

Semantic segmentation requires capturing multi-scale context. Standard CNNs lose resolution through pooling. DeepLab v3 uses atrous (dilated) convolutions to enlarge receptive field without losing resolution.

## Core Idea

A ResNet backbone extracts features. ASPP applies parallel atrous convolutions with different dilation rates (6, 12, 18) to capture multi-scale context. Results are concatenated and upsampled to produce pixel-level predictions.

## Architecture

### Overview

![deeplabv3 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x512x512 |
| 2 | ResNet Backbone | `conv2d` | dilated ResNet-101 |
| 3 | ASPP | `conv2d` | rates 6/12/18 + global pooling |
| 4 | Concat + Decoder | `identity` | concat + bilinear upsample |
| 5 | Segmentation Map | `output` | per-pixel class labels |

</details>

### Components

1. **Atrous convolution** — dilated convolution enlarges receptive field without losing resolution. 2. **ASPP** — parallel atrous convolutions at rates 6, 12, 18 + 1x1 conv + global average pooling. 3. **ResNet backbone** — dilated ResNet-101 with output stride 16. 4. **Bilinear upsampling** — upsample feature map to input resolution. 5. **Multi-scale context** — captures objects at different scales.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A ResNet backbone extracts features. ASPP applies parallel atrous convolutions with different dilation rates (6, 12, 18)...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DeepLab v2 (ASPP), FCN. Successor: DeepLab v3+ (encoder-decoder), SegFormer.

## References

- Chen et al. 2017
