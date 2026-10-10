# Architecture: Instant-NGP

## Motivation

NeRF training takes hours due to the cost of querying a large MLP millions of times. Instant-NGP uses a multiresolution hash grid for spatial encoding, enabling real-time training (seconds to minutes).

## Core Idea

Replace the positional encoding with a multiresolution hash table. The network is tiny (2-4 layers, 64 units). The hash table stores learnable features at multiple resolutions, enabling O(1) spatial lookups.

## Architecture

### Overview

![instant-ngp architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Coordinate (x,y,z) | `input` | shape: [3] |
| 2 | Hash Encoding | `embedding` | multiresolution, L=16 levels, F=2 features |
| 3 | Concat | `identity` | concatenate features from all levels |
| 4 | Tiny MLP | `linear` | outFeatures: 64, inFeatures: 32 (2 hidden layers) |
| 5 | ReLU | `relu` |  |
| 6 | Output MLP | `linear` | outFeatures: 4 (density + RGB) |
| 7 | Density + Color | `output` |  |

</details>

### Components

1. **Multiresolution hash encoding** — L resolution levels, each with a hash table of size T. For each level, vertices of the grid cell containing the query point are looked up, and features are trilinearly interpolated. 2. **Hash collision resolution** — Collisions are resolved by training; the network learns to handle collisions gracefully. 3. **Tiny MLP** — Only 2 hidden layers of 64 units, since the hash encoding does the heavy lifting. 4. **Fully fused MLP** — Custom CUDA kernels that fuse all MLP operations into a single kernel. 5. **Exponentially spaced levels** — Grid resolution grows exponentially from coarse to fine.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace the positional encoding with a multiresolution hash table. The network is tiny (2-4 layers, 64 units). The hash table stores learnable features at multiple resolutions, enabling O(1) spatial l
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NeRF. Successor: Plenoxels, K-Planes, merf.

## References

- Muller et al. 2022
