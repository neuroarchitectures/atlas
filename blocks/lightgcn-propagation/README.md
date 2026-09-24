# LightGCN Propagation

## Design Philosophy

GCN's feature transforms and nonlinearities are trainable complexity without proven benefit for collaborative filtering — embeddings live on a simple manifold. Strip the GCN layer to *pure neighborhood propagation* (no W, no σ) and take the mean over all layers as the final embedding.

## Functionality

- `e^{(k+1)}_u = Σ_{v∈N(u)} (1/√(|N(u)||N(v)|)) · e^{(k)}_v` — symmetric-normalized aggregation only.
- Final: `e_u = mean_k e^{(k)}_u` (layer-combination; no layer weights).

## Used By

| Model | Role |
|-------|------|
| LightGCN | 3 propagation layers, 64-dim, layer-mean over 4 embeddings |

## Features

- **Fewer parameters, better ranking** — the empirical core of the paper.
- **Layer mean = adaptive hop mixing** without learned selection.

## Evolution

- **Predecessor**: GCN (graph-convolution-layer), NGCF.
- **Related**: mix-hop propagation (MTGNN) adds decoupled hop selection when graphs are directed; APPNP's personalized PageRank propagation.
