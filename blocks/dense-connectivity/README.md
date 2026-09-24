# Dense Connectivity Block (DenseNet)

## Design Philosophy

Instead of ResNet's additive skip (one-to-one), connect **every layer to every other layer in its block** by concatenation. Each layer receives the concatenation of all preceding feature maps and adds only a small "growth" of new channels. The philosophy: *feature reuse* — don't re-learn features you already have, just concatenate them. This gives strong accuracy at very few parameters and improves gradient flow.

## Functionality

A dense block of L layers, each: `BN-ReLU-1×1 → BN-ReLU-3×3`, output `k` channels (the growth rate).

- Layer `l`'s input = `concat(x_0, x_1, ..., x_{l-1})` → `k·l` input channels.
- Layer `l`'s output = `x_l` → `k` new channels appended.
- **Transition layers** between dense blocks: BN-1×1-conv + 2×2 avg-pool to halve channels and spatial resolution.

## Used By

| Model | Role |
|-------|------|
| DenseNet-121 | 4 dense blocks (6/12/24/16 layers), growth rate 32 |
| DenseNet-169 / 201 | Deeper dense blocks |
| DenseNet-BC | Bottleneck-compressed variant |

## Features

- **Feature reuse**: Direct access to all prior features, no re-learning.
- **Parameter efficiency**: Each layer adds only `k` channels; a 121-layer net has 8M params.
- **Gradient flow**: Short paths from loss to every layer.
- **Memory cost**: Concatenation grows the feature map, which is why transitions compress it.

## Evolution

- **Predecessor**: ResNet (additive skip, one-to-one) — DenseNet generalizes to all-to-all.
- **Successor**: Mostly superseded by ResNet-style architectures for practical reasons (memory), but the concat-skip idea lives on in U-Net and FPN.
