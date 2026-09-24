# Non-Linear Tuplewise Similarity

## Design Philosophy

Some hyperedge functions are *provably indecomposable* — no combination of pairwise similarities can express them. DHNE therefore scores the whole tuple with a deep non-linear classifier over typed embeddings, trained jointly with a neighborhood-reconstruction autoencoder that preserves the graph structure the tuples came from.

## Functionality

- Type-specific embeddings per node type; tuple (e.g., ⟨user, item, tag⟩) → deep net → similarity score.
- Joint loss: tuplewise classification + autoencoder reconstruction of neighborhood structure.

## Used By

| Model | Role |
|-------|------|
| DHNE | Hyperedge embedding on heterogeneous hypergraphs (3-node tuples) |

## Features

- **Indecomposability-motivated architecture** — theory directly dictates the design.
- **Joint structure preservation** — the AE keeps local topology in the embedding.

## Evolution

- **Predecessor**: pairwise-factorized hyperedge scoring.
- **Related**: hyperedge-position-embedding (slot transforms instead of deep scoring).
