# Continuous-Filter Convolution (SchNet)

## Design Philosophy

In a molecule, the influence of one atom on another should depend smoothly on their distance. A discrete convolution has no notion of continuous geometry. SchNet replaces the fixed filter bank with filters *generated* from the interatomic distance, so the layer is smooth and rotationally/translationally invariant.

## Functionality

For each atom `i`, aggregate over neighbours `j` with a distance-dependent filter:

```
x_i' = sum_{j != i} x_j * W( ||r_i - r_j|| )
```

where `W(d)` comes from a filter-generating network applied to a radial-basis expansion of `d`:

```
W(d) = MLP( rbf(d) )
```

The same filter is shared across neighbours, and the operation depends only on distances — never on absolute orientation.

## Used By

SchNet

## Features

- **Continuous**: outputs vary smoothly with atomic positions.
- **Invariant**: uses distances only, so rotations/translations leave it unchanged.
- **Shared filter MLP**: one network generates all filters.
- **Composable**: stacked interaction blocks build many-body effects.

## Evolution

Predecessor: brute-force Coulomb / hand-crafted potentials, discrete filters. Successor: direction-aware variants (DimeNet, directional message passing), equivariant MPNNs (PaiNN, NequIP).
