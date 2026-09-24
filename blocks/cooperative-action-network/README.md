# Cooperative Action-Environment GNN

## Design Philosophy

Message passing treats all nodes uniformly; CoGNN lets each node *choose its own communication behavior*. A jointly trained action network predicts per-node, per-layer actions over {STANDARD, LISTEN, BROADCAST, ISOLATE}; the sampled actions induce a fresh directed computational graph that the environment network actually runs.

## Functionality

- Action network π: per node per layer, action distribution over 4 communication actions.
- Sampled actions (Gumbel straight-through) define the message topology for environment network η; decouples input graph from computational graph.

## Used By

| Model | Role |
|-------|------|
| CoGNN | Wraps any base GNN (GCN/GIN/GAT); actions resampled every layer |

## Features

- **Learned per-node computation graph** — heterophily and homophily handled locally.
- **Expressivity + efficiency dial** — ISOLATE/BROADCAST give fine-grained sparsity control.

## Evolution

- **Predecessor**: fixed-topology GNNs; graph rewiring (DIGL).
- **Related**: variance-reduced-gcn-sampling — sampling for scalability, not behavior.
