# Set Abstraction (PointNet++)

## Design Philosophy

Hierarchical feature learning for point clouds. Each set abstraction layer samples a subset of points, groups their neighbors, and applies PointNet to each group, creating a hierarchy of local regions.

## Functionality

1. Farthest Point Sampling (FPS): select n' points maximally spread out. 2. Ball query: find all points within radius r around each sampled point. 3. PointNet: apply shared MLP + max pooling to each group.

## Used By

PointNet++ | PointCNN | Hierarchical point cloud models

## Features

- **Hierarchical**: Creates multi-scale local features.
- **FPS sampling**: Ensures spatial coverage.
- **Ball query**: Local neighborhood extraction.

## Evolution

Predecessor: PointNet (global). Successor: Vector attention (Point Transformer), edge conv (DGCNN).
