# Architecture: SchNet

## Motivation

Molecules have variable numbers of atoms and continuous interatomic distances. Standard CNNs can't handle this. SchNet introduces continuous-filter convolutions where filter weights are generated from interatomic distances.

## Core Idea

Each atom is represented by an embedding. Continuous-filter convolutions use distance-dependent filter-generating networks to model pairwise interactions. Multiple interaction blocks refine atom representations. A pooling readout predicts molecular properties.

## Architecture

### Overview

![schnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | N atoms x 3 coords |
| 2 | Atom Embedding | `linear` | element type -> 64 dim |
| 3 | Continuous-Filter Conv | `conv1d` | distance-dependent filters |
| 4 | Interaction Block | `linear` | 6 blocks, CFConv + dense |
| 5 | Readout | `linear` | atom-wise + sum pooling |
| 6 | Molecular Property | `output` | energy, forces, etc. |

</details>

### Components

1. **Atom embeddings** — learned embeddings per element type. 2. **Filter-generating network** — MLP that takes interatomic distance and outputs filter weights. 3. **Continuous-filter convolution** — applies distance-dependent filters to atom features. 4. **Interaction blocks** — stacked CFConv + dense layers (typically 6). 5. **Readout** — atom-wise dense + sum pooling to molecular property. 6. **Rotation/translation equivariant** — respects physical symmetries.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Each atom is represented by an embedding. Continuous-filter convolutions use distance-dependent filter-generating networ...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DTNN (distance-based), MPNN. Successor: DimeNet, GemNet, PaiNN (directional/equivariant).

## References

- Schütt et al. 2017
