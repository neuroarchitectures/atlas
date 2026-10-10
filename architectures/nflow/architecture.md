# Architecture: Neural Splines Flow

## Motivation

Affine coupling flows (RealNVP) can only scale and shift, limiting expressiveness. Neural spline flows use monotonic rational-quadratic splines as transform functions, enabling much more flexible density estimation.

## Core Idea

Split input into two parts. A neural network conditioned on one part generates a monotonic rational-quadratic spline that transforms the other part. Stack multiple coupling + permutation layers. Exact log-likelihood via change of variables.

## Architecture

### Overview

![nflow architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Data | `input` | D-dim |
| 2 | Split Channels | `identity` | split into 2 halves |
| 3 | Neural Spline Coupling | `linear` | rational-quadratic spline |
| 4 | Permutation | `identity` | reverse channels |
| 5 | Latent | `output` | Gaussian latent |

</details>

### Components

1. **Rational-quadratic splines** — monotonic, invertible, flexible. 2. **Coupling layers** — condition spline on one half, transform other. 3. **Neural network conditioner** — NN outputs spline parameters. 4. **Exact likelihood** — change of variables gives exact log-likelihood. 5. **Stacked** — multiple coupling + permutation layers.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Split input into two parts. A neural network conditioned on one part generates a monotonic rational-quadratic spline tha...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: RealNVP (affine), Glow. Successor: Flow Matching, Rectified Flow.

## References

- Durkan et al. 2019
