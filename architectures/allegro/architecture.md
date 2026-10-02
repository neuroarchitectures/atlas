# Architecture: Allegro

## Motivation

Message-passing GNNs for interatomic potentials require sequential neighbor communication, limiting GPU parallelism for large systems.

## Core Idea

Use a purely local equivariant architecture (no message passing) that computes per-atom features from local neighbor geometry, enabling efficient parallel computation.

## Architecture

### Overview

![allegro architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | shape: [20, 3] |
| 2 | Atom Embedding | `embedding` | dim: 128 |
| 3 | Local MLP | `mlp` |  |
| 4 | Tensor Product | `tensor-product` |  |
| 5 | Output MLP | `mlp` |  |
| 6 | Global Pooling | `pooling` |  |
| 7 | Energy | `output` |  |

</details>

### Components

1. **Atom embedding** — Per-element learnable embedding. 2. **Local feature extraction** — MLP on pairwise distances and directions. 3. **Equivariant tensor product** — Combine scalar and vector features using Clebsch-Gordan products. 4. **Per-atom energy** — MLP predicts per-atom energy contribution. 5. **Sum** — Total energy is sum of per-atom energies.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a purely local equivariant architecture (no message passing) that computes per-atom features from local neighbor geo
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NequIP. Successor: EquiformerV2.

## References

Musaelian et al. 2023
