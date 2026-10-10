# Point Transformer Attention

## Design Philosophy

Apply attention to point clouds. Each point attends to its k nearest neighbors, with attention weights computed from relative position encoding.

## Functionality

For each point i and neighbor j: gamma(phi(q_i) - psi(k_j) + delta(p_i - p_j)). The position encoding delta uses relative coordinates. The aggregation is a weighted sum of value features.

## Used By

Point Transformer | Point cloud segmentation | 3D scene understanding

## Features

- **Position-aware**: Uses relative position encoding.
- **Local attention**: Only attends to k nearest neighbors.
- **Continuous**: Works directly on unordered point sets.

## Evolution

Predecessor: PointNet++, EdgeConv. Successor: Point Transformer V2/V3.
