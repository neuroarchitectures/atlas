# Architecture: PaiNN

## Motivation

Standard GNNs for molecular property prediction are not rotationally equivariant, limiting generalization to molecular orientations.

## Core Idea

Use equivariant message passing that maintains both scalar and vector features, enabling SE(3)-equivariant predictions without the overhead of full spherical harmonics.

## Architecture

### Overview

![painn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | shape: [20, 3] |
| 2 | Atom Embedding | `embedding` | dim: 128 |
| 3 | Interaction Block | `message-passing` |  |
| 4 | Interaction Block | `message-passing` |  |
| 5 | Interaction Block | `message-passing` |  |
| 6 | Global Pooling | `pooling` |  |
| 7 | Property Head | `linear` | outFeatures: 1 |
| 8 | Molecular Property | `output` |  |

</details>

### Components

1. **Atom embedding** — Learnable embeddings per atom type. 2. **Interaction blocks** — Message passing with both scalar and vector features. 3. **Equivariant updates** — Vector features updated using direction-dependent messages. 4. **Global pooling** — Sum or mean over atoms. 5. **Property head** — Linear layer predicts molecular property.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use equivariant message passing that maintains both scalar and vector features, enabling SE(3)-equivariant predictions w
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SchNet, DimeNet. Successor: Equiformer, NequIP.

## References

Schütt et al. 2021
