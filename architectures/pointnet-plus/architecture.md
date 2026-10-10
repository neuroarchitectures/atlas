# Architecture: PointNet++

## Motivation

PointNet's global max pooling loses local geometric structure. PointNet++ introduces hierarchical feature learning by recursively applying PointNet to local neighborhoods.

## Core Idea

Each Set Abstraction (SA) layer: (1) sample points using farthest point sampling, (2) group neighboring points within a ball, (3) apply PointNet to each group. This creates a hierarchy of local regions.

## Architecture

### Overview

![pointnet++ architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Point Cloud | `input` | shape: [1024, 3] |
| 2 | Set Abstraction 1 | `identity` | FPS sample 512, group, PointNet |
| 3 | Set Abstraction 2 | `identity` | FPS sample 128, group, PointNet |
| 4 | Set Abstraction 3 | `identity` | FPS sample 64, group, PointNet |
| 5 | Feature Propagation | `identity` | interpolate features back to all points |
| 6 | Per-Point Labels | `output` | segmentation |

</details>

### Components

1. **Farthest Point Sampling (FPS)** — Selects a subset of points that are maximally spread out, ensuring coverage. 2. **Ball query** — Groups all points within a radius around each sampled point. 3. **PointNet per group** — A shared PointNet processes each group, producing local features. 4. **Feature Propagation (FP)** — For segmentation, features are interpolated from coarser to finer layers. 5. **Multi-scale grouping (MSG)** — Groups at multiple radii for multi-scale context.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Each Set Abstraction (SA) layer: (1) sample points using farthest point sampling, (2) group neighboring points within a ball, (3) apply PointNet to each group. This creates a hierarchy of local region
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: PointNet. Successor: PointCNN, DGCNN, PointTransformer.

## References

- Qi et al. 2017
