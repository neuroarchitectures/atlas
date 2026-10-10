# Zero Convolution (ControlNet)

## Design Philosophy

When injecting a new conditioning signal into a pre-trained diffusion model, naively connecting the condition destroys the pretrained features and forces retraining. Initialising the injection convolution to zero makes the module start as an identity — the base model is unchanged at step 0 and the new signal is learned in gradually.

## Functionality

A `1x1` convolution whose weight and bias are initialised to zero, placed on every connection between the trainable copy and the frozen backbone:

```
y = Conv_{1x1}(x)      with W = 0, b = 0
```

At initialisation `y = 0`, so the residual addition leaves the network identical to the pretrained model. Gradients flow and the weights grow smoothly during training.

## Used By

ControlNet

## Features

- **Identity at init**: no disruption to pretrained weights.
- **Stable training**: the condition is faded in, not blasted in.
- **One per residual connection**: scales to many injection points.
- **Zero-parameter overhead at init**: adds capacity only as it learns.

## Evolution

Predecessor: fine-tuning the whole backbone, adapter layers. Sibling: gated cross-attention (zero-init scalar gate). Successor: zero-init conditioning in ControlNet variants and adapters (T2I-Adapter).
