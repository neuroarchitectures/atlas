# Multilayer Structural Similarity Graph

## Design Philosophy

Structural role should be scale-aware: two hubs are similar globally, two bridges locally. struc2vec builds an auxiliary *multilayer graph* where layer k connects nodes by exp(−d_k(u,v)) — DTW distance between k-hop degree sequences — and cross-layer self-edges with weight γ let biased random walks *choose the comparison scale* as they walk.

## Functionality

- Layer k: intra-layer edges weighted by structural similarity of k-hop degree sequences (DTW-aligned).
- Cross-layer edges connect the same node at adjacent layers; walks (biased by similarity, γ for layer transitions) feed skip-gram — walking the auxiliary graph, not the original network.

## Used By

| Model | Role |
|-------|------|
| struc2vec | Structural-role embeddings robust to degree mismatch |

## Features

- **Scale-selective walks** — the walk length distribution over layers tunes role granularity.
- **No dependence on node identity** — purely structural equivalence.

## Evolution

- **Predecessor**: RolX/RoleX role discovery; graphlet counting.
- **Related**: graphlet-degree-vector (discrete role fingerprints); deep symmetric embeddings.
