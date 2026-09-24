# FCN Mask Branch

## Design Philosophy

Mask prediction needs per-pixel, not per-box, structure — so give each RoI a *fully convolutional* branch that emits an m×m mask per class directly, decoupled from the box/classification branch. Class-specific masks with sigmoid (not cross-class softmax) let classes compete through instance separation, not mask channels.

## Functionality

- RoIAlign features → 4×(conv 3×3, stride 1/2/1/1) → deconv upsample → 1×1 conv → K m×m sigmoid masks.
- Loss on positive RoIs only (L_mask on the GT class channel); separate from box regression.

## Used By

| Model | Role |
|-------|------|
| Mask R-CNN | Parallel mask head alongside box + class heads on 14×14/28×28 RoI features |

## Features

- **Pixel-to-pixel alignment** — enabled by RoIAlign's exact quantization-free sampling.
- **Decoupled from classification** — mask quality doesn't suffer from box-head interference.

## Evolution

- **Predecessor**: FCIS, MNC (position-sensitive masks).
- **Successor**: PointRend, Mask2Former (masked attention + queries); prototype-mask-head (YOLACT) as the dynamic-filter alternative.
