# Architecture: Glow

## Motivation

Earlier normalizing flows (RealNVP, NICE) had limited expressiveness. Glow introduced invertible 1x1 convolutions to generalize channel-wise permutations, improving flow capacity.

## Core Idea

Stack three types of invertible transformations: ActNorm (per-channel scale and bias), Invertible 1x1 Convolution (learned channel mixing), and Affine Coupling Layers (split-and-transform).

## Architecture

### Overview

![glow architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [3, 256, 256] |
| 2 | ActNorm | `batchNorm` | per-channel scale+bias (data-dependent init) |
| 3 | Invertible 1x1 Conv | `conv` | kernelSize: 1, learned channel permutation |
| 4 | Affine Coupling | `conv` | split channel, transform half conditioned on other |
| 5 | ActNorm 2 | `batchNorm` |  |
| 6 | Output | `output` | exact log-likelihood |

</details>

### Components

1. **ActNorm** — Per-channel learnable scale and bias, initialized from the first batch's mean and std. Replaces BatchNorm for invertible flows. 2. **Invertible 1x1 Conv** — A 1x1 convolution with a square weight matrix W; the log-determinant is log|det(W)|, computed in O(n^3) where n is channels. 3. **Affine coupling** — Split channels into two halves; transform one half with s, t conditioned on the other: x_a' = x_a, x_b' = (x_b + t(x_a)) * exp(s(x_a)).

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Stack three types of invertible transformations: ActNorm (per-channel scale and bias), Invertible 1x1 Convolution (learned channel mixing), and Affine Coupling Layers (split-and-transform).
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: RealNVP, NICE. Successor: Neural Spline Flows, Flow++.

## References

- Kingma & Dhariwal 2018
