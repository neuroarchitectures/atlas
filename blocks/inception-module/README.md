# Inception Module

## Design Philosophy

Instead of committing to one kernel size, run **multiple filter sizes in parallel** and concatenate the results. The philosophy: at each layer, you don't know whether a 1×1, 3×3, or 5×5 filter is the right choice, so let the network learn to use all of them. The 1×1 convs in each branch serve as dimension reduction (bottleneck) before the expensive 3×3/5×3 convs, controlling cost.

## Functionality

Four parallel branches, outputs concatenated on the channel axis:

1. **1×1 conv** — captures cross-channel interactions at a point.
2. **1×1 → 3×3 conv** — local spatial patterns.
3. **1×1 → 5×5 conv** — wider spatial patterns.
4. **3×3 max-pool → 1×1 conv** — pooled features.

All branches output the same spatial size (via padding), so concatenation works. The 1×1 bottlenecks reduce channels before the spatial convs, keeping FLOPs manageable.

## Used By

| Model | Role |
|-------|------|
| GoogLeNet / Inception-v1 | The defining module, 9 inception blocks |
| Inception-v2 / v3 | Factorized convs (n×1 + 1×n instead of n×n) |
| Inception-ResNet | Adds residual connections to inception |
| TimesNet | 2D inception block for time-series period analysis |

## Features

- **Multi-scale**: Different kernel sizes capture features at different spatial scales simultaneously.
- **Sparsity**: The architecture is locally sparse (only selected paths fire), approximating a dense layer more efficiently.
- **Bottleneck 1×1s**: The key trick that keeps the parameter count from exploding.

## Evolution

- **Predecessor**: Network-in-Network (NiN, 2013) — introduced 1×1 convs as cross-channel MLPs.
- **Successor**: Inception-v3 factorizes convs; Inception-ResNet adds skips; ultimately superseded by ResNet's simpler design for most tasks.
