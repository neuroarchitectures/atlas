# Architecture: EGNN

## Motivation

SE(3)-Transformers are computationally expensive due to Clebsch-Gordan products. EGNN achieves E(n) equivariance with simple operations, making it scalable to large graphs.

## Core Idea

EGNN updates node features and coordinates separately. Coordinate updates use relative positions (which are translation-invariant) and message passing. No spherical harmonics needed.

## Architecture

### Overview

![egnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Nodes + Coords | `input` | shape: [100, 6] (h, x) |
| 2 | EGCL 1 | `identity` | Equivariant Graph Conv Layer |
| 3 | EGCL 2 | `identity` | Equivariant Graph Conv Layer |
| 4 | EGCL 3 | `identity` | Equivariant Graph Conv Layer |
| 5 | Equivariant Features | `output` |  |

</details>

### Components

1. **Equivariant Graph Conv Layer (EGCL)** — Updates both node features h and coordinates x. 2. **Coordinate update** — x_i' = x_i + sum_j (1/||x_i-x_j||+1) * phi_x(h_i, h_j, ||x_i-x_j||) * (x_i - x_j). This is E(n)-equivariant because it uses relative positions. 3. **Feature update** — h_i' = h_i + sum_j phi_h(h_i, h_j, ||x_i-x_j||, a_ij). Uses squared distances (invariant) as edge features. 4. **No spherical harmonics** — The key insight: using relative positions directly achieves equivariance without the computational overhead of irreducible representations.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — EGNN updates node features and coordinates separately. Coordinate updates use relative positions (which are translation-invariant) and message passing. No spherical harmonics needed.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GNN, SE(3)-Transformer. Successor: Equiformer, MACE.

## References

- Satorras et al. 2021
