# Directional Message Passing (DimeNet)

## Design Philosophy

In molecules, bond *angles* carry as much chemistry as bond lengths. Message passing that only uses pairwise distances cannot distinguish bond angles. DimeNet passes messages along directed edges and expands the angle between incoming and outgoing bonds into a spherical-harmonic basis.

## Functionality

Messages flow on directed edges `(j -> i)`. The message depends on both the distance `d_ji` and the angle `angle_kji` between incoming edge `(k -> j)` and outgoing edge `(j -> i)`:

```
m_ji = sum_{k != j} f( d_ji, rbf(d_kj), angle_kji )
```

Angles are expanded with spherical harmonics `Y^{(l)}` and combined via a 2-D (spherical + radial) Fourier-Bessel basis; distance terms use Bessel functions.

## Used By

DimeNet

## Features

- **Angle-aware**: distinguishes molecular geometry that distances alone cannot.
- **Directional edges**: messages are carried on directed bonds.
- **Spherical + radial basis**: joint angular/radial expansion.
- **Rotation invariant**: uses only distances and angles.

## Evolution

Predecessor: distance-only message passing (SchNet). Successor: GemNet (adds dihedral/torsion), equivariant directional models (PaiNN, NequIP).
