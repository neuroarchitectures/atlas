# SE(3)-Equivariant Attention

## Design Philosophy

Attention that respects 3D rigid transformations. Keys, queries, values are typed by SE(3) representation order. Attention weights are scalars (invariant), ensuring equivariance.

## Functionality

1. Embed features as irreducible representations (l=0,1,2,...). 2. Compute attention weights from scalar (l=0) features. 3. Aggregate value features using Clebsch-Gordan tensor products. 4. Output is SE(3)-equivariant.

## Used By

SE(3)-Transformer | Equiformer | Molecular property prediction

## Features

- **Exact equivariance**: Outputs rotate/translate with inputs.
- **Clebsch-Gordan products**: Combine representations.
- **Spherical harmonics**: Basis for angular features.

## Evolution

Predecessor: Tensor Field Networks. Successor: Equiformer, MACE.
