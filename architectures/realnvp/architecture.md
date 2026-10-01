# Architecture: RealNVP

## Motivation

Earlier normalizing flows (NICE) used additive coupling layers with limited expressiveness. RealNVP introduced affine coupling layers that can scale and shift, significantly improving generation quality.

## Core Idea

Split the input tensor along the channel dimension into two halves. Transform one half with an affine function conditioned on the other half. The Jacobian of this transformation is triangular, making its determinant trivial to compute.

## Architecture

### Overview

![realnvp architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [3, 32, 32] |
| 2 | Affine Coupling 1 | `conv` | split channels, affine transform on half |
| 3 | Affine Coupling 2 | `conv` | reverse split pattern |
| 4 | Affine Coupling 3 | `conv` | checkerboard masking pattern |
| 5 | Output | `output` | exact log-likelihood |

</details>

### Components

1. **Affine coupling layer** — Splits input x into (x1, x2). Computes s, t = NN(x1). Then y1 = x1, y2 = (x2 + t) * exp(s). The Jacobian is triangular: det = prod(exp(s)). 2. **Masking patterns** — Uses checkerboard and channel-wise masks alternately to ensure all dimensions are transformed. 3. **Multi-scale architecture** — Stacks coupling layers with squeezing operations (spatial-to-channel reshaping) at different resolutions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Split the input tensor along the channel dimension into two halves. Transform one half with an affine function conditioned on the other half. The Jacobian of this transformation is triangular, making 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NICE (additive coupling). Successor: Glow, Neural Spline Flows.

## References

- Dinh et al. 2017
