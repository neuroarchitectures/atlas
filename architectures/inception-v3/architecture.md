# Architecture: Inception v3

## Motivation

Inception v1/v2 used large filters. Inception v3 factorizes large convolutions into smaller ones (7x7 → 1x7 + 7x1) for efficiency, adds label smoothing and batch normalization for regularization.

## Core Idea

Stem network with factorized convolutions. Inception modules with asymmetric factorized convolutions (n×1 + 1×n). Auxiliary classifiers for gradient flow. Label smoothing, BN-auxiliary, and RMSProp for training stability.

## Architecture

### Overview

![inception-v3 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x299x299 |
| 2 | Stem | `conv2d` | factorized convolutions |
| 3 | Inception Blocks | `conv2d` | 11 modules |
| 4 | Auxiliary Classifier | `conv2d` | 2 aux heads |
| 5 | Classification | `output` | 1000 classes |

</details>

### Components

1. **Factorized convolutions** — 7x7 → 1x7 + 7x1, reducing params. 2. **Asymmetric inception** — modules with n×1 and 1×n branches. 3. **Label smoothing** — soft targets for regularization. 4. **BN-auxiliary** — batch norm in auxiliary classifiers. 5. **Grid-size reduction** — efficient downsampling without pooling. 6. **RMSProp** — optimizer choice for stability.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Stem network with factorized convolutions. Inception modules with asymmetric factorized convolutions (n×1 + 1×n). Auxili...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Inception v1 (GoogLeNet), v2. Successor: Inception v4, Inception-ResNet.

## References

- Szegedy et al. 2016
