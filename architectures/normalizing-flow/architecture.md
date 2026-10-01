# Architecture: Normalizing Flow

## Motivation

VAEs and GANs either approximate or implicitly define distributions. Normalizing flows provide exact likelihood computation by using invertible transformations with tractable Jacobian determinants.

## Core Idea

Start with a simple base distribution (e.g., Gaussian) and apply a sequence of invertible transformations. The change-of-variables formula gives exact log-likelihood: log p(x) = log p(z) + log|det(df^{-1}/dz)|.

## Architecture

### Overview

![normalizing flow architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Sample z | `input` | shape: [2] — base distribution |
| 2 | Flow Layer 1 | `linear` | Planar flow: f(z) = z + u*h(w^T z + b) |
| 3 | Flow Layer 2 | `linear` | Radial flow: f(z) = z + b*alpha/(r+|z-z0|) |
| 4 | Flow Layer 3 | `linear` | Planar flow |
| 5 | Flow Layer 4 | `linear` | Radial flow |
| 6 | Sample x | `output` | log p(x) = log p(z) + log|det J| |

</details>

### Components

1. **Planar flow** — f(z) = z + u * h(w^T z + b), where h is a smooth elementwise nonlinearity. The Jacobian determinant is 1 + u^T psi(w^T z + b) w. 2. **Radial flow** — f(z) = z + alpha * beta * (z - z0) / (alpha + ||z - z0||). 3. **Change of variables** — log p(x) = log p(z) - log|det(df/dz)|, computed exactly due to invertibility.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Start with a simple base distribution (e.g., Gaussian) and apply a sequence of invertible transformations. The change-of-variables formula gives exact log-likelihood: log p(x) = log p(z) + log|det(df^
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NICE, RealNVP. Successor: Glow, Neural Spline Flows.

## References

- Rezende & Mohamed 2015
