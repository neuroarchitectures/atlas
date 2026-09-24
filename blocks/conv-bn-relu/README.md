# Conv-BN-ReLU Block

## Design Philosophy

The most fundamental convolutional building block: a 2D convolution followed by BatchNorm followed by a ReLU activation. The philosophy is **linear transform → normalize → nonlinearity**, the three-step recipe that made deep CNNs trainable. By inserting BatchNorm between the conv and the activation, internal covariate shift is tamed, higher learning rates become usable, and the saturating-gradient problem of tanh/sigmoid is replaced by ReLU's non-saturating gradient.

## Functionality

- **Conv2D**: learns spatial filters that slide over the input, mixing channels and spatial neighborhoods. Parameters: `outChannels`, `kernelSize`, `stride`, `padding`, `inChannels`.
- **BatchNorm**: normalizes each channel to zero mean / unit variance across the batch, with learnable scale (γ) and shift (β). Stabilizes training.
- **ReLU**: `max(0, x)` — non-saturating activation that prevents vanishing gradients and enables faster training than sigmoid/tanh.

Typical data flow: `input → conv2d → batchNorm → relu → output`.

## Used By

| Model | Role |
|-------|------|
| VGG-16 | Every conv layer follows conv-ReLU (pre-BN era used raw conv-ReLU; modern ports add BN) |
| ResNet-50 | Inside each residual block (conv-BN-ReLU ×2) |
| AlexNet | Conv-ReLU (original used LRN instead of BN; BN is the modern replacement) |
| U-Net | Encoder/decoder conv layers |
| Mask R-CNN | Backbone conv stages |

## Features

- **Trainability**: BatchNorm enables 10×+ learning rates and deeper networks.
- **Modularity**: Stack N copies to build arbitrarily deep CNNs.
- **Inference fusion**: Conv+BN can be fused into a single conv at deploy time (BN scale folded into conv weights).
- **Spatial locality**: Each conv sees only a local receptive field; stacking grows it linearly.

## Evolution

- **Predecessor**: Raw conv + ReLU (AlexNet, 2012) — no normalization, relied on careful initialization and LRN.
- **Successor**: Residual block wraps two of these + a skip connection (ResNet, 2015).
- **Modern variant**: ConvNeXt replaces BatchNorm with LayerNorm and ReLU with GeLU, questioning whether BN was essential.
