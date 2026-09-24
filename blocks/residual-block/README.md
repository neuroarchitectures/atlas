# Residual Block (Identity Skip)

## Design Philosophy

The block that solved the degradation problem: as networks get deeper, plain networks get *harder* to train (not because of vanishing gradients, but because learning the identity mapping through stacked nonlinear layers is difficult). The residual block reframes the layer's job: instead of learning `H(x)`, learn the residual `F(x) = H(x) − x`. The identity skip connection carries `x` forward unchanged, so the layers only need to learn the *correction*. If the optimal transform is close to identity, the block just drives `F → 0`.

## Functionality

Structure: `input → conv-BN-ReLU → conv-BN → add(input) → ReLU → output`.

- Two 3×3 convs (same channel count, stride 1) form the residual function `F(x)`.
- An element-wise `add` node merges `F(x)` with the identity `x`.
- The final ReLU is post-addition.
- When dimensions change (stride 2 or channel change), the skip uses a 1×1 conv projection instead of identity.

## Used By

| Model | Role |
|-------|------|
| ResNet-50 | Basic block (the 50-layer uses bottleneck variants; ResNet-34 uses this exact form) |
| ResNet-18 / ResNet-34 | The plain two-conv residual unit |
| Wide ResNet | Wider versions of the same block |
| Every modern Transformer | The residual stream concept descends directly from this idea |

## Features

- **Identity initialization**: A stack of residual blocks initialized at zero behaves as the identity — training starts from a good baseline.
- **Gradient highway**: The skip connection gives gradients a direct path backward, enabling 100+ layer training.
- **Universal influence**: The residual stream is the backbone of every Transformer block (`x + Sublayer(Norm(x))`).

## Evolution

- **Predecessor**: Highway networks (Srivastava et al., 2015) — gating instead of identity.
- **Successor**: Bottleneck residual (1×1-3×3-1×1) for deeper/wider ResNets.
- **Conceptual descendant**: Transformer residual stream, U-Net skip connections (different: concat, not add).
