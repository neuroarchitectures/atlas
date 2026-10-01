# Architecture: Kolmogorov-Arnold Network

## Motivation

MLPs have fixed activation functions at nodes. The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be represented as a composition of univariate functions. KANs put learnable splines on edges.

## Core Idea

Replace linear layers + fixed activations with learnable B-spline activation functions on edges. Each edge has a learnable univariate function (B-spline), and nodes simply sum their inputs. This is more parameter-efficient for certain tasks.

## Architecture

### Overview

![kolmogorov-arnold network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [2] |
| 2 | KAN Layer 1 | `linear` | outFeatures: 5, inFeatures: 2 (B-spline on edges) |
| 3 | KAN Layer 2 | `linear` | outFeatures: 1, inFeatures: 5 |
| 4 | Output | `output` |  |

</details>

### Components

1. **Learnable edge functions** — Each edge has a B-spline function phi_{i,j}(x) with learnable control points. 2. **Node aggregation** — Nodes simply sum their incoming edge functions: y_i = sum_j phi_{i,j}(x_j). 3. **Kolmogorov-Arnold theorem** — f(x1,...,xn) = sum_i Phi_i(sum_j phi_{i,j}(x_j)), a 2-layer composition of univariate functions. 4. **SiLU residual** — Each edge function has a SiLU residual: phi(x) + SiLU(x) * w_b for stability. 5. **Interpretability** — Individual edge functions can be visualized, making KANs more interpretable than MLPs.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace linear layers + fixed activations with learnable B-spline activation functions on edges. Each edge has a learnable univariate function (B-spline), and nodes simply sum their inputs. This is mo
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MLP. Related: Spline networks, KAN variants (FastKAN, EfficientKAN).

## References

- Liu et al. 2024
