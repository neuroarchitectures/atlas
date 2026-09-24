# Tensorized Hypergraph Convolution

## Design Philosophy

A k-uniform hyperedge couples k nodes at once — pairwise incidence matrices lose that. THNN represents hyperedges as a k-dimensional adjacency tensor with *high-order outer-product message passing*, then tames the O(d^N) cost by partially symmetric CP decomposition (→ O(Nd)), with scalar-1 lower-order terms and Tanh stabilization.

## Functionality

- Tensor adjacency Θ over k-tuples; message = high-order outer product of node embeddings within each hyperedge.
- CP decomposition with partially symmetric factors; per-layer Θ transforms; low-order correction terms.

## Used By

| Model | Role |
|-------|------|
| THNN | k-uniform hypergraph learning (e.g., 3-uniform object hypergraphs) |

## Features

- **True higher-order interactions** — not pairwise approximations.
- **CP compression** — the only thing making order-N feasible.

## Evolution

- **Predecessor**: HGNN (spectral hypergraph convolution on incidence matrix).
- **Related**: multi-arity-hyperedge-layer (expand-reduce over arities), hyperedge-position-embedding (scoring-side).
