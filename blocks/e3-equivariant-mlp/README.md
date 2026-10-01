# E(3)-Equivariant MLP

## Design Philosophy

An MLP that respects E(3) symmetry (rotations, translations, reflections). Achieved by separating scalar (l=0) and vector (l=1) features and using equivariant operations.

## Functionality

Node features are typed by representation order. Scalar channels use standard MLP. Vector channels use relative position-based updates. Clebsch-Gordan products combine channels of different orders.

## Used By

EGNN | e3nn library | MACE

## Features

- **No spherical harmonics**: Uses relative positions directly.
- **Simple**: Much simpler than SE(3)-Transformer.
- **Scalable**: O(N^2) for N nodes, practical for large graphs.

## Evolution

Predecessor: GNN. Successor: MACE, EquiformerV2.
