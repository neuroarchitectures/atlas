# Multi-View Graph Fusion

## Design Philosophy

A network has many simultaneous truths: multiple edge types (multiplex), multiple semantic graph views (neighbor/function/transport), multiple proximity definitions (topology vs properties), multiple modalities. Embed each view separately, then *fuse adaptively* — per-node, learned weights decide which view matters where.

## Functionality

- Per view g: embedder → view embedding `X^(g)`; fusion: `x_u = Σ_g α_{u,g} · X^(g)_u` with learned (attention-style) per-node weights.
- Variants: MHGCN aggregates across edge types and hops with adaptive importance; MGCN fuses GCNs over multiple semantic graphs per timestep; MVC-DNE cross-view autoencoders supervise one view from another; FMTSF two-phase attention first within modality, then across modalities.

## Used By

| Model | Role |
|-------|------|
| MHGCN | Multiplex heterogeneous GNN: depth (hops) + breadth (edge types) with adaptive weights |
| ST-MGCN | GCN over 3 semantic graph views (neighborhood/function/transport), fused per step |
| MVNE | Attention-based per-node view fusion of proximity-view embeddings |
| DNE-PL (MVC-DNE) | Cross-view autoencoder learning between topology and property views |
| FMTSF | Two-phase modality attention over price/event/KG sources |

## Features

- **Adaptive per-node weighting** — different regions can live in different views.
- **View-specific encoders** — heterogeneous semantics handled natively.

## Evolution

- **Predecessor**: single-view GNNs; HAN's semantic-level attention.
- **Related**: multi-gate-moe (task-side view fusion); cgc-expert-separation (expert views).
