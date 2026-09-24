# Antisymmetric Edge-Local Frame GNN

## Design Philosophy

Physical laws are symmetric (SO(3) equivariance) and antisymmetric (Newton's third law: F_ij = −F_ji). Build these in by construction: scalarize node vectors onto an edge-local orthonormal frame (a, b, c), decode edge embeddings into *antisymmetric* forces and angular interactions, and share a reference point — the network cannot violate the physics it's given.

## Functionality

- Per edge: local frame from relative position; node vectors scalarized onto the frame; edge decoder outputs antisymmetric F_ij, A_ij plus shared reference terms.
- 6-DoF: init block + shared block, sub-time-stepped rollout for dynamics prediction.

## Used By

| Model | Role |
|-------|------|
| PI-GNN | Physics-informed interaction prediction with built-in equivariance/antisymmetry |

## Features

- **Physics by architecture** — symmetry constraints are structural, not loss penalties.
- **Exact momentum conservation** from antisymmetry.

## Evolution

- **Predecessor**: SE(3)-Transformer, EGNN (equivariant but not pairwise-antisymmetric).
- **Related**: NequIP/MACE — higher-order equivariant message passing.
