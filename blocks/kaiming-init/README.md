# Kaiming (He) Initialization

## Design Philosophy

Initialization should **preserve the variance of activations and gradients through a layer**, not just at the start. Kaiming init derives the required variance from the forward/backward variance propagation of a **ReLU-family** (or more generally, rectifier) nonlinearity — accounting for the fact that ReLU zeroes half of its input, which doubles the variance you must inject.

## Functionality

Weights drawn from `N(0, 2 / fan_in)` (forward-preserving) or `U(−√(6/fan_in), √(6/fan_in))`; PyTorch's `kaiming_normal_`/`kaiming_uniform_` take a `nonlinearity` argument (`relu`, `leaky_relu` with `a`).

- `fan_in` = `kernel_height · kernel_width · in_channels` for a conv layer.
- Biases are typically initialized to zero; the final classification layer often uses a smaller scale.
- Usually paired with BatchNorm — with BN in every conv, the init matters less, but it still sets the early-training scale.

## Used By

| Model | Role |
|-------|------|
| ResNet / ResNeXt / ResNet-based DETRs | Conv weight init |
| CNN backbones (FPN, YOLO, RTMPose CSPNeXt) | Conv weight init |
| U-Net / segmentation decoders | Conv weight init |
| Any ReLU-family CNN without normalization | Essential for 30+ layer nets |

## Features

- **Variance-preserving for rectifiers**: factor of 2 vs. Xavier accounts for ReLU's zeroing.
- Works **both forward and backward** (two modes, `fan_in` / `fan_out`).
- No learned parameters, no extra compute — pure initialization scheme.

## Evolution

- **Predecessors**: small random normal init (pre-2010, fragile), Xavier/Glorot init (2010, linear/tanh assumption).
- **Itself**: He et al., 2015 (Delving Deep into Rectifiers).
- **Successors/variants**: Fixup init (train very deep ResNets without norm), T-Fixup (transformers without LayerNorm), LayerScale, and scaled/truncated init used by ViT and GPT-2 (`N(0, 0.02)` truncated normal).
