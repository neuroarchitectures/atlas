# U-Net Encoder-Decoder with Skip Connections

## Design Philosophy

A symmetric encoder-decoder where **long skip connections** concatenate encoder features into the matching decoder level. The philosophy: the contracting (encoder) path captures *context* (what), the expanding (decoder) path recovers *localization* (where), and the skips carry the high-resolution detail that the bottleneck would otherwise lose. This shape is still the default for dense prediction a decade later.

## Functionality

- **Encoder**: Repeated `conv3×3-BN-ReLU → 2×2 max-pool` (halve spatial, double channels).
- **Bottleneck**: Conv layers at the lowest resolution.
- **Decoder**: Repeated `transpose-conv2×2 (upsample 2×) → concat(encoder_skip) → conv3×3-BN-ReLU`.
- **Skip connections**: Encoder level `i`'s pre-pool features are concatenated into decoder level `i` (channel-wise).
- **Output**: 1×1 conv to the desired channel count (e.g., class count).

## Used By

| Model | Role |
|-------|------|
| U-Net | Biomedical segmentation (the original) |
| Diffusion U-Net (Stable Diffusion) | The noise predictor, with cross-attention added |
| Mask R-CNN | Mask head is a mini U-Net |
| Many segmentation models | The default backbone topology |

## Features

- **Localization via skips**: Without the concat-skips, the decoder cannot recover fine spatial detail.
- **Symmetric**: Encoder and decoder mirror each other, so feature shapes align for concatenation.
- **No fully-connected layers**: Fully convolutional, works at any input resolution.

## Evolution

- **Predecessor**: FCN (fully convolutional networks for segmentation).
- **Successor**: Diffusion U-Net (adds cross-attention + timestep conditioning); U-Net++ (nested/dense skips); ultimately DiT shows a Transformer can replace the U-Net for diffusion.
