# Mutual Selective Attention

## Design Philosophy

Two linked entities (an edge, a pair of documents) describe each other. Mutual attention lets each side's attention weights *guide the other's encoder*: u attends to v's words guided by what v attends to in u — producing context-dependent, per-edge embeddings instead of static node features.

## Functionality

- Per edge (u,v): word-level encoders (CNN/LSTM) for both sequences; co-attention matrix exchanges emphasis; outputs u(v) and v(u) concatenated with the static vertex embedding.

## Used By

| Model | Role |
|-------|------|
| CANE | Context-aware network embedding; per-edge representations fed to DeepWalk/LINE-style objectives |

## Features

- **Edge-conditioned embeddings** — one node, many context-specific views.
- **Bidirectional guidance** — symmetric information exchange between the pair.

## Evolution

- **Predecessor**: static node embeddings (DeepWalk/LINE/node2vec).
- **Related**: cross-attention (one-directional conditioning); target-attention-din (query-to-sequence specialization).
