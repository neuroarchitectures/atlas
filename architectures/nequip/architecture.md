# Architecture: NequIP

## Motivation

Learning interatomic potentials requires SE(3)-equivariance. Standard GNNs use invariant features and lose directional information.

## Core Idea

Use E(3)-equivariant convolutions with spherical harmonics and tensor products, enabling the network to learn directional interactions while maintaining equivariance.

## Architecture

### Overview

![nequip architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | shape: [20, 3] |
| 2 | Atom Embedding | `embedding` | dim: 32 |
| 3 | Equivariant Conv | `message-passing` |  |
| 4 | Equivariant Conv | `message-passing` |  |
| 5 | Equivariant Conv | `message-passing` |  |
| 6 | Global Pooling | `pooling` |  |
| 7 | Energy Head | `linear` | outFeatures: 1 |
| 8 | Total Energy | `output` |  |

</details>

### Components

1. **Atom embedding** — One-hot or learned embeddings per element. 2. **Equivariant convolutions** — Spherical harmonic-based message passing with tensor products. 3. **Clebsch-Gordan tensor products** — Combine features of different angular momentum. 4. **Global pooling** — Sum over atoms for total energy. 5. **Energy head** — Scalar prediction.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use E(3)-equivariant convolutions with spherical harmonics and tensor products, enabling the network to learn directiona
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SchNet, DimeNet, PaiNN. Successor: Allegro, Equiformer.

## References

Batzner et al. 2022
