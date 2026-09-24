# Structural Re-Parameterization

## Design Philosophy

Multi-branch training topologies (3×3 + 1×1 + identity) are good for optimization but bad for latency. Keep the multi-branch block *only during training*; at deploy time algebraically fold all branches into a single 3×3 convolution — identical function, single-branch inference topology.

## Functionality

- Train: `y = Conv3x3(x) + Conv1x1(x) + x` (BN in each branch).
- Deploy: merge BN into convs, pad 1×1 and identity kernels to 3×3, sum kernels into one `Conv3x3`.

## Used By

| Model | Role |
|-------|------|
| RepViT | RepViT blocks (3×3 + branch paths folded pre-latency), MobileNetV3-derived stem/stages |
| YOLO-family detectors | RepConv fusion blocks in necks |

## Features

- **Train-time capacity, deploy-time speed** — no accuracy/latency trade-off at inference.
- **Exact equivalence** — pure linear algebra, not distillation or approximation.

## Evolution

- **Predecessor**: RepVGG (origin); multi-branch blocks like residual-block and inception-module.
- **Successor**: RepOptimizer (directly training the re-parameterized form); DBB, MobileOne variants.
