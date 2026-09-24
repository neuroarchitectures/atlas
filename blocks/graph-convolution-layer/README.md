# Graph Convolution Layer (GCN)

## Design Philosophy

Spectral graph convolution via a **localized first-order approximation**. The philosophy: instead of expensive eigendecompositions (Bruna, 2014) or Chebyshev expansions (Defferrard, 2016), simplify to a single propagation rule that aggregates each node's normalized neighborhood features through a learned linear transform. The graph structure is encoded directly into the model `f(X, A)`, trained end-to-end.

## Functionality

Propagation rule: `H^(l+1) = σ(Â H^(l) W^(l))`, where:
- `Ã = A + I_N` (add self-loops)
- `Â = D̃^{-1/2} Ã D̃^{-1/2}` (renormalized adjacency, precomputed once)
- `W^(l)` is a layer-specific weight matrix
- `σ` is ReLU

A two-layer GCN: `Z = softmax(Â ReLU(Â X W^(0)) W^(1))`.

## Used By

| Model | Role |
|-------|------|
| GCN (Kipf & Welling) | The defining model |
| GraphSAGE (mean aggregator) | Nearly identical propagation rule |
| DGI (Deep Graph Infomax) | GCN as the encoder |
- The propagation rule became the foundation for the "message passing" framework.

## Features

- **O(|E|) complexity**: Scales to graphs with millions of edges (sparse matmul).
- **End-to-end**: Structure + features trained jointly, no separate embedding pipeline.
- **Shared weights**: One weight matrix per layer, shared across all nodes.
- **Transductive**: Best when the graph is fixed; inductive generalization needs GraphSAGE.

## Evolution

- **Predecessor**: Spectral GNNs (Bruna, 2014, O(N²)); ChebNet (Defferrard, 2016, K-order).
- **Successor**: GraphSAGE (inductive, sampling); GAT (attention-based weights); FastGCN (mini-batch).
