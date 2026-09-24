# Graph Propagation Attention (GPA)

## Design Philosophy

Graph transformers bolt a GNN alongside attention with separate FFNs — duplicated machinery. GPA defines a *single* attention module with three propagation paths: node→node (global self-attention + learned edge bias), node→edge, and edge→node (dynamic product weights), updating nodes and edges together without a second FFN stream.

## Functionality

- Node-node: standard MHSA with layer-specific edge bias φ(e_ij).
- Node-edge: propagate through A and softmax(A) row/col-normalized expansions.
- Edge-node: self-product dynamic weights; edge embeddings updated every block; virtual node carries global signal.

## Used By

| Model | Role |
|-------|------|
| GraphProp | Transformer blocks with virtual node; FFN expansion α=1 |

## Features

- **One module, three routes** — removes the dual-FFN redundancy of hybrid graph transformers.
- **Edges as first-class tokens** — updated, not just used as bias.

## Evolution

- **Predecessor**: Graphormer (edge bias), GPS (hybrid MPNN+attention with dual FFNs).
- **Related**: edge-feature-attention — the minimal edge-injection variant.
