# Architecture: DimeNet

## Motivation

SchNet uses only interatomic distances, losing angular information. Molecular interactions depend on bond angles. DimeNet adds directional message passing using both distances and angles.

## Core Idea

Atom embeddings are updated using messages that depend on both interatomic distances and angles. Bessel basis functions expand distances, spherical harmonics expand angles. Directional message passing propagates information along directional edges.

## Architecture

### Overview

![dimenet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atoms + Geometry | `input` | atom types + 3D positions |
| 2 | Atom Embedding | `linear` | element -> 128 dim |
| 3 | Distance + Angle Embedding | `identity` | Bessel + spherical harmonics |
| 4 | Directional Message Passing | `linear` | 6 interaction blocks |
| 5 | Readout | `linear` | sum pooling |
| 6 | Molecular Property | `output` | energy, HOMO-LUMO, etc. |

</details>

### Components

1. **Atom embeddings** — per-element-type embeddings. 2. **Bessel basis functions** — expand interatomic distances. 3. **Spherical harmonics** — expand interatomic angles. 4. **Directional message passing** — messages flow along directional edges (atom pairs). 5. **Interaction blocks** — stacked message passing + dense layers. 6. **Readout** — sum pooling to molecular property.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Atom embeddings are updated using messages that depend on both interatomic distances and angles. Bessel basis functions ...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SchNet (distances only). Successor: GemNet (full geometric), PaiNN (equivariant).

## References

- Gasteiger et al. 2020
