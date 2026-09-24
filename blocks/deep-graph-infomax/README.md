# Deep Graph Infomax (DGI)

## Design Philosophy

Node labels are scarce; mutual information between local patches and the global graph summary is free supervision. Encode nodes, read out a graph summary, corrupt by shuffling node features, and train a bilinear discriminator to tell real (node, summary) pairs from corrupted ones — maximizing local-global MI.

## Functionality

- `D(h_i, s) = σ(h_iᵀ W s)`; positive pairs from the true graph, negatives from feature-shuffled graph.
- Readout: mean pool (or attention pool); loss: binary cross-entropy / Jensen-Shannon over positive and negative scores.

## Used By

| Model | Role |
|-------|------|
| DGI | Unsupervised node embeddings over GCN/GraphSAGE encoders |

## Features

- **No negative node sampling** — corruption is feature shuffling, not random pairs.
- **Transfers the InfoNCE idea** to graphs with a global summary term.

## Evolution

- **Predecessor**: InfoNCE-contrastive-loss (CPC).
- **Successor**: GRACE/GCA graph contrastive learning with augmentations; BGRL asymmetric variants.
