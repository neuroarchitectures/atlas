# Architecture: SE(3)-Transformer

## Motivation

Standard transformers are not equivariant to 3D rotations, requiring data augmentation. SE(3)-Transformer embeds SE(3) equivariance directly, guaranteeing consistent behavior under rigid transformations.

## Core Idea

Use irreducible representations of the SE(3) group (scalar, vector, higher-order tensors). Attention is computed in an SE(3)-equivariant way using tensor field convolutions and Clebsch-Gordan products.

## Architecture

### Overview

![se(3)-transformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | 3D Points + Features | `input` | shape: [100, 6] (xyz + features) |
| 2 | SE(3) Embedding | `linear` | irreducible representations (scalars, vectors) |
| 3 | SE(3) Attention | `attention` | Clebsch-Gordan tensor product based |
| 4 | Equivariant FFN | `linear` | non-linear equivariant MLP |
| 5 | Equivariant Output | `output` | rotates/translates with input |

</details>

### Components

1. **Irreducible representations** — Features are typed by SE(3) representation order (l=0 scalar, l=1 vector, l=2 tensor). 2. **Tensor field convolution** — Neighborhood aggregation using radial basis functions and Clebsch-Gordan coefficients. 3. **SE(3)-equivariant attention** — Keys, queries, values are SE(3)-typed; attention weights are scalars (l=0) for invariance. 4. **Clebsch-Gordan product** — Combines representations to produce higher/lower order representations.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use irreducible representations of the SE(3) group (scalar, vector, higher-order tensors). Attention is computed in an SE(3)-equivariant way using tensor field convolutions and Clebsch-Gordan products
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Tensor Field Networks, Transformer. Successor: Equiformer, e3nn library.

## References

- Fuchs et al. 2020
