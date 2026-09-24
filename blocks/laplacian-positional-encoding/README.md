# Laplacian Positional Encoding

## Design Philosophy

Graphs have no canonical order, so "position" must mean *structural position*. The graph Laplacian's smallest non-trivial eigenvectors are exactly the smooth, globally-consistent coordinates of a graph — project k of them per node and add at input: a structural position signal with no learned PE table.

## Functionality

- Eigendecompose L = I − D^{-1/2} A D^{-1/2}; take k smallest non-trivial eigenvectors per node.
- Sign ambiguity (eigenvectors ±) handled by random flips at training; linear projection → added to node features at layer 0 only.

## Used By

| Model | Role |
|-------|------|
| Graph Transformer | Input-only Laplacian PE (k eigenvectors) alongside standard attention |

## Features

- **Canonical structural coordinates** — distance in eigenspace ≈ diffusion distance.
- **Input-only injection** — no per-layer cost beyond the initial projection.

## Evolution

- **Predecessor**: random-walk PEs (Dwivedi & Bresson), degree features.
- **Related**: spectral graph embeddings (grarep's PPMI-SVD is the factorization cousin); SAN's learnable PE.
