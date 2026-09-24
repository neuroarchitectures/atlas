# ConvNeXt Block

## Design Philosophy

Take a Transformer block and replace the attention with a **large-kernel depthwise conv**. The philosophy: question how much of the Vision Transformer's win was "attention" vs. "training recipe + architecture tricks." By modernizing a pure ConvNet with every Transformer-era trick (LayerNorm, GeLU, inverted bottleneck, large 7×7 kernels, fewer activations), ConvNeXt matches Swin on ImageNet — without any attention.

## Functionality

Structure: `input → depthwise 7×7 → LayerNorm → 1×1 expand(4×) → GeLU → 1×1 project → add(input)`.

1. **Depthwise 7×7 conv**: Large-kernel spatial mixing (the "attention replacement").
2. **LayerNorm**: Not BatchNorm — the Transformer-style norm.
3. **1×1 expand → GeLU → 1×1 project**: An inverted-bottleneck FFN, exactly like a Transformer FFN.
4. **Residual**: Identity skip around the whole block.
5. Only one activation (GeLU) and one norm — minimal, Transformer-style.

## Used By

| Model | Role |
|-------|------|
| ConvNeXt-Tiny/Small/Base/Large | 4 stages, depths 3/3/9/3, the block stacked per stage |
| ConvNeXt-V2 | Same block + FCMAE self-supervised pretraining |

## Features

- **No attention**: Pure conv, yet tracks Swin Transformer.
- **Large kernels**: 7×7 depthwise — the spatial receptive field ViT gets from attention.
- **Transformer aesthetics**: LayerNorm, GeLU, inverted bottleneck, sparse activations.

## Evolution

- **Predecessor**: ResNet bottleneck; Vision Transformer (the block mimics a Transformer block).
- **Successor**: ConvNeXt-V2 (FCMAE masked-autoencoder pretraining); ongoing research on conv-vs-attention.
