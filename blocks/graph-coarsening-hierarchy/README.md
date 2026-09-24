# Graph Coarsening Hierarchy

## Design Philosophy

Embedding a 10M-node graph directly is expensive and noisy at the fine scale. Coarsen the graph level by level (HARP: star clustering + edge collapse; MILE: hybrid Structural-Equivalence + Normalized Heavy Edge Matching), embed the *coarsest* graph, then uncoarsen embeddings level by level — each level warm-started by the coarser one, MILE refining via a learned GCN through the matching matrices.

## Functionality

- HARP: recursive coarsening (~50% per level, L = 3–5), embeddings propagated down as initializations; wraps DeepWalk/LINE/node2vec unchanged.
- MILE: `A_{i+1} = Mᵀ A_i M` coarsening; refinement `E_i = GCN(E_{i+1}, A_i, M)` back up the hierarchy.

## Used By

| Model | Role |
|-------|------|
| HARP | Hierarchical warm-started embedding of large graphs |
| MILE | Graph embedding via coarsening + GCN-based refinement |

## Features

- **Any base embedder** — the hierarchy is an orthogonal wrapper.
- **Quality + speed** — coarse structure resolved cheaply, fine detail refined locally.

## Evolution

- **Predecessor**: Louvain-style coarsening for community detection; GraSSEP.
- **Related**: multi-resolution-parallel-streams (HRNet) — the opposite direction (preserve fine scale throughout).
