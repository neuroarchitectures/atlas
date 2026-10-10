# PointNet Shared MLP

## Design Philosophy

Apply the same MLP to each point independently. This is permutation-invariant (since the same function is applied to all points) and can be implemented as a 1x1 convolution.

## Functionality

For each point (x, y, z, ...), apply a shared linear layer: h_i = W * x_i + b. The same W and b are used for all points. Multiple layers can be stacked with non-linearities.

## Used By

PointNet | PointNet++ | DGCNN (with modifications)

## Features

- **Permutation invariant**: Same function for all points.
- **1x1 conv equivalent**: Can be implemented as a 1x1 convolution.
- **Per-point processing**: No inter-point interaction.

## Evolution

Predecessor: MLP. Successor: Edge conv (DGCNN), point transformer attention.
