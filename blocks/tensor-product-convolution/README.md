# Tensor Product Convolution (TFN)

## Design Philosophy

A molecule's scalar features (energies, charges) and vector features (forces, dipoles) live in different representations of the rotation group. To mix them while staying equivariant, combine them with the same algebra physics uses: the tensor product and Clebsch-Gordan decomposition.

## Functionality

Features are irreps of `SO(3)` with angular momentum `l` (scalars `l=0`, vectors `l=1`, ...). A convolution multiplies a feature with a spherical-harmonic-weighted neighbour direction and reduces the product back to irreps:

```
(L1 ⊗ L2)_Lm = sum_{m1, m2} C_{m1 m2 m}^{L1 L2 L} f^{L1}_{m1} g^{L2}_{m2}
```

where `C` are Clebsch-Gordan coefficients. The result transforms correctly under rotation by construction.

## Used By

TFN | NequIP | Allegro

## Features

- **Exact equivariance**: never breaks `SO(3)` symmetry.
- **Multi-channel irreps**: mixes scalars, vectors, and higher tensors.
- **Clebsch-Gordan basis**: physics-grounded multiplication rule.
- **Learnable radial weights**: geometry enters through radial networks.

## Evolution

Predecessor: invariant message passing (SchNet). Successor: `e3nn`-style tensor products, `E(3)`-equivariant ViT/attention, higher-order equivariant MPNNs.
