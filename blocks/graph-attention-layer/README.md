# Graph Attention Layer (GAT)

## Design Philosophy

Graph convolution with **masked self-attention** that computes dynamic, data-driven edge weights. The philosophy: GCN uses fixed, structure-defined weights (the normalized adjacency); GAT learns to assign different importance to different neighbors through a shared attention mechanism. This decouples the model from a specific graph structure, enabling inductive generalization to unseen graphs.

## Functionality

1. **Shared linear transform**: `W ∈ R^{F'×F}` applied to every node: `Wh_i`.
2. **Attention coefficients**: `e_ij = LeakyReLU(ã^T [Wh_i || Wh_j])` for each edge `(i, j)`.
3. **Masked attention**: Only compute `e_ij` for first-order neighbors `j ∈ N_i` (inject graph structure).
4. **Softmax normalization**: `α_ij = softmax_j(e_ij)` — comparable across nodes with varying degrees.
5. **Aggregation**: `h'_i = σ(Σ_{j∈N_i} α_ij W h_j)`.
6. **Multi-head**: K independent heads; concatenate on hidden layers, average on output.

## Used By

| Model | Role |
|-------|------|
| GAT (Veličković et al.) | The defining model |
| Various GNN variants | Attention-based message passing |

## Features

- **Dynamic edge weights**: Data-driven, unlike GCN's fixed normalization.
- **Inductive**: Applicable to unseen graphs (no eigendecomposition, no structure dependence).
- **Multi-head**: K heads stabilize learning, like Transformer MHA.
- **No costly matrix ops**: No eigendecomposition or inversion.

## Evolution

- **Predecessor**: GCN (fixed weights); spectral GNNs (structure-dependent).
- **Successor**: Graph Transformer (full attention over nodes); various attention-based GNN variants.
