# Architecture: PointNet

## Motivation

Point clouds are unordered sets — standard CNNs require regular grids. PointNet processes raw points directly using shared MLPs and symmetric max pooling for permutation invariance.

## Core Idea

Apply shared MLP to each point independently, then max-pool across all points to get a global feature vector. The global feature is concatenated with per-point features for segmentation.

## Architecture

### Overview

![pointnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Point Cloud | `input` | shape: [1024, 3] (N points, xyz) |
| 2 | Shared MLP 1 | `linear` | outFeatures: 64, inFeatures: 3 (per-point) |
| 3 | ReLU | `relu` |  |
| 4 | Shared MLP 2 | `linear` | outFeatures: 1024, inFeatures: 64 |
| 5 | Max Pool | `identity` | symmetric, permutation-invariant |
| 6 | Global Feature | `linear` | outFeatures: 512 |
| 7 | Classification | `output` |  |

</details>

### Components

1. **Shared MLP** — Same weights applied to each point independently (1x1 conv equivalent). 2. **Max pooling** — Symmetric function that aggregates N points into a single global feature, ensuring permutation invariance. 3. **T-Net** — A mini-PointNet that predicts an affine transformation matrix to align the input point cloud. 4. **Feature transform** — Concatenate global feature with per-point features for segmentation tasks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Apply shared MLP to each point independently, then max-pool across all points to get a global feature vector. The global feature is concatenated with per-point features for segmentation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: VoxNet (voxelized). Successor: PointNet++, PointCNN, DGCNN.

## References

- Qi et al. 2017
