# GraphSAGE Neighborhood Aggregator

## Design Philosophy

Learn **aggregator functions** that generate embeddings by sampling and aggregating features from a node's local neighborhood, rather than training a distinct embedding per node. The philosophy: this enables inductive generalization to unseen nodes — no embedding lookup is needed, and the learned function applies to any node with features. Inspired by the Weisfeiler-Lehman isomorphism test.

## Functionality

For each search depth `k = 1...K`:
1. **Sample neighbors**: Draw a fixed-size uniform sample `N(v)` from node `v`'s neighbors.
2. **Aggregate**: `h_{N(v)}^k = AGGREGATE_k({h_u^{k-1}, ∀u ∈ N(v)})` — mean, LSTM, or pooling.
3. **Concatenate + transform**: `h_v^k = σ(W^k · CONCAT(h_v^{k-1}, h_{N(v)}^k))`.
4. **L2 normalize**: `h_v^k = h_v^k / ||h_v^k||_2`.

Aggregator variants:
- **Mean**: Elementwise mean of neighbor vectors.
- **LSTM**: LSTM on a random permutation of neighbors.
- **Pooling**: `max({σ(W_pool h_u + b), ∀u ∈ N(v)})` — symmetric, trainable.

## Used By

| Model | Role |
|-------|------|
| GraphSAGE | The defining model |
| PinSAGE | Pinterest's production version (Random-walk sampling) |
- The aggregator framework underlies most inductive GNNs.

## Features

- **Inductive**: Generates embeddings for unseen nodes/graphs in a single forward pass.
- **Fixed compute**: Sampling keeps per-batch cost fixed at `O(∏ S_i)`.
- **Flexible aggregator**: Mean/LSTM/pooling, pluggable.

## Evolution

- **Predecessor**: GCN (transductive, full neighborhood); DeepWalk/node2vec (per-node embeddings).
- **Successor**: GAT (attention-based aggregation); PinSAGE (production-scale sampling); various inductive GNNs.
