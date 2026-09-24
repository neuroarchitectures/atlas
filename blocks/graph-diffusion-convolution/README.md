# Convolution-Based Graph Diffusion

## Design Philosophy

Local message passing reaches only k-hop neighbors — but functionally related regions may be far apart in the graph. Diffuse information through the graph (multi-step heat-kernel-style propagation) so signals reach distant but similar nodes, complementing local attention filters.

## Functionality

- Diffusion operator over powers of the (normalized) adjacency — `Σ_k θ_k T^k`-style propagation reaching remote nodes with learned per-step weights.
- Used alongside GAT attention in the hierarchical spatial module.

## Used By

| Model | Role |
|-------|------|
| ST-GDN | Hierarchical spatial module: diffusion conv + local GAT attention over region graphs |

## Features

- **Reaches functionally-similar distant nodes** — beyond strict hop neighborhoods.
- **Diffusion weights learnable per step** — the graph analog of dilated convolutions.

## Evolution

- **Predecessor**: GCN fixed k-hop aggregation; Diffusion-Convolutional Neural Networks (Atwood & Towsley).
- **Related**: PPR/GDC preprocessing; mix-hop propagation (MTGNN).
