# Invariant Point Attention (IPA)

## Design Philosophy

Protein structure prediction must output 3-D coordinates while staying equivariant to global rotation and translation. Standard attention operates on invariant scalar features and cannot reason about geometry. IPA folds the 3-D point positions directly into the attention mechanism so the layer is `SE(3)`-equivariant by construction.

## Functionality

Each residue carries a local frame and a point in space. IPA computes attention weights from three terms and combines them:

```
a_ij = softmax( w_L * (q_i . k_j)                         (scalar)
              - w_C * gamma_ij * ||T_i . q_i - T_j . k_j||^2  (point)
              + w_D * ||T_i . q_i - T_j . k_j||^2 )        (pair)
```

Scalar and point messages are summed in the local frame, then projected back — so the output points transform correctly under any rigid motion of the whole structure.

## Used By

AlphaFold 2 | ESMFold

## Features

- **SE(3)-equivariant**: rotation/translation of the input rotates/translates the output.
- **Geometry-aware**: attends over Euclidean distances between points, not just features.
- **Local frames**: uses backbone frames to define the point coordinate system.
- **Frames-update**: outputs per-residue rigid frames for the structure module.

## Evolution

Predecessor: egde/attention over distance features (GNN potentials). Sibling: SE(3)-Transformer. Successor: IPA variants in single-sequence folders (ESMFold) and other structure predictors.
