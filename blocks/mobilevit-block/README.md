# MobileViT Block

## Design Philosophy

Use a transformer *as a convolution*: unfold local CNN features into a sequence of overlapping patches, run a compact transformer to encode global information, then fold back into feature-map shape. The result injects global receptive fields while preserving the CNN's spatial inductive bias and learnability on small data.

## Functionality

- Local reps: `ConvNxN → Conv1x1` per block.
- Unfold feature map into non-overlapping patches → transformer encoder (MHSA + FFN) → fold back → `Conv1x1` fusion + `ConvNxN` merge with local path.

## Used By

| Model | Role |
|-------|------|
| MobileViT | Core block after conv stems; ~6M-parameter backbone reaching ViT-level accuracy |

## Features

- **Transformers as convolutions** — implicit positional info via unfold/fold, no explicit PE needed.
- **Mobile-friendly** — far fewer attention FLOPs than patch-token ViTs.

## Evolution

- **Predecessor**: patch-embedding ViT (global attention from the first layer, data-hungry).
- **Successor**: MobileViT v2 (separable self-attention for linear cost); hybrid stacks like MobileNetV4 and MaxViT.
