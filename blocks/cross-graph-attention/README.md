# Cross-Graph Attention Matching

## Design Philosophy

Graph similarity is more than node similarity — it's relational consistency. Cross-graph attention computes how each node in graph 1 matches nodes in graph 2, and feeds the *difference vector* `a_{j→i}(h_i − h_j)` into node updates: on a perfect match the signal vanishes, on mismatch it amplifies — training the model to be sensitive to discrepancy.

## Functionality

- Per round: cross-attention weights between graphs; matched-difference messages added to GRU node updates; T rounds of propagation; O(|V₁||V₂|).
- Graph-level gated readout (sigmoid-gated weighted sum) feeds the similarity scorer.

## Used By

| Model | Role |
|-------|------|
| GMN (Graph Matching Networks) | Pairwise graph similarity scoring; siamese + cross-graph interaction |

## Features

- **Discrepancy-amplifying updates** — the mechanism is most active exactly when graphs differ.
- **Learned, not hand-crafted alignment** — no fixed node correspondence needed.

## Evolution

- **Predecessor**: Siamese graph encoders (independent embedding + distance).
- **Successor**: subgraph matching variants; attention-based matching inside retrieval systems.
