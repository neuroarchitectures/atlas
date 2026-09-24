# Graphlet Degree Vector (GDV)

## Design Philosophy

A node's structural role is the *shapes it participates in*, not just its degree. Count occurrences of each 2–5-node graphlet (induced subgraph) centered at the node: the resulting vector is a structural fingerprint capturing structural equivalence exactly, usable alone or concatenated with random-walk embeddings.

## Functionality

- For each node: orbit-level counts over graphlets up to size 5 (up to 73 orbits).
- Combined with word2vec-style walk embeddings (joint or concatenated) for the final representation.

## Used By

| Model | Role |
|-------|------|
| SNE-Enhanced | Structural fingerprints + random-walk embeddings, jointly optimized |

## Features

- **Exact structural equivalence** — the count vector is a canonical role signature.
- **Interpretable dimensions** — each orbit is a named structural motif.

## Evolution

- **Predecessor**: degree distributions; motif counting (Milo et al. 2002).
- **Related**: struc2vec's structural-similarity-multigraph — continuous distance version of the same idea.
