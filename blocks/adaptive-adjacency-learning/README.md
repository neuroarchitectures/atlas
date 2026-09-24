# Adaptive Adjacency Learning

## Design Philosophy

Many domains have no trustworthy graph, or the "true" relations are hidden. Learn the adjacency from data: embed nodes, and construct the graph from embedding similarity — softmax(ReLU(E₁E₂ᵀ)) (MTGNN), top-k similarity (GAD), attention over learned embeddings (StemGNN), or uncertainty-weighted thresholds (UAGSL) — trained jointly with the downstream task.

## Functionality

- MTGNN: `A = softmax(ReLU(E1 E2ᵀ))`, sparsified by top-k per node; directed graph supported.
- Variants: GAD top-k neighbor selection; StemGNN self-attention latent correlation (recomputed per forward); UAGSL entropy-based per-node uncertainty thresholds on directional edges.

## Used By

| Model | Role |
|-------|------|
| MTGNN | Graph learning layer feeding mix-hop GCN for multivariate forecasting |
| GAD | Top-k sensor similarity graph for anomaly detection |
| StemGNN | Latent correlation layer replacing pre-defined inter-series topology |
| UAGSL | Uncertainty-aware structure learning on embedding-based GSL |

## Features

- **Graph as a parameter** — topology adapts with the task, no domain graph required.
- **Sparsification controls** — top-k / thresholding keep the learned graph usable.

## Evolution

- **Predecessor**: fixed adjacency GCN; NRI (neural relational inference).
- **Successor**: latent graph inference (LG-AL); diffusion-generated graphs.
