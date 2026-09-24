# Topological Message Passing (Simplicial/Cell Complexes)

## Design Philosophy

Graphs model only pairwise adjacency; real systems have triangles, cavities, and higher-order boundaries. TDL propagates on *simplicial/cell complexes* via algebraic operators — adjacency (within dimension), incidence, boundary/coboundary, and Hodge Laplacian (across dimensions) — so messages flow both within and across topological levels.

## Functionality

- Within-dimension: message passing over the k-skeleton adjacency.
- Cross-dimension: boundary ∂ and coboundary ∂* operators move signals between k-cells and (k±1)-cells; Hodge Laplacian L_k = (∂∂*)-type operators define the spectral view.

## Used By

| Model | Role |
|-------|------|
| TDL (Topological Deep Learning) | Message passing on simplicial/cell complexes across dimensions |

## Features

- **Higher-order structure native** — holes and cavities are representable.
- **Algebraic grounding** — operators, not heuristics, define neighborhood.

## Evolution

- **Predecessor**: simplicial neural networks (Ebli et al.), CW networks.
- **Related**: multi-arity-hyperedge-layer — the hypergraph (set-based) cousin.
