# Architecture: Cluster-GCN

## Motivation

Training GCNs on large graphs is memory-intensive because full-batch GCN requires the entire graph's adjacency matrix in memory.

## Core Idea

Partition the graph into clusters using a graph clustering algorithm (METIS), then process each cluster as a mini-batch, enabling scalable training on large graphs.

## Architecture

### Overview

![cluster-gcn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | shape: [3703] |
| 2 | GCN Layer 1 | `gcn-layer` | outChannels: 500 |
| 3 | ReLU | `relu` |  |
| 4 | GCN Layer 2 | `gcn-layer` | outChannels: 100 |
| 5 | ReLU | `relu` |  |
| 6 | Classifier | `linear` | outFeatures: 10 |
| 7 | Node Classification | `output` |  |

</details>

### Components

1. **Graph clustering** — METIS partitions the graph into p clusters. 2. **Cluster batching** — Each mini-batch is a cluster subgraph. 3. **GCN layers** — Standard graph convolution within each cluster. 4. **Hierarchical clustering** — Merge small clusters for better utilization.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Partition the graph into clusters using a graph clustering algorithm (METIS), then process each cluster as a mini-batch,
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GraphSAGE, FastGCN. Successor: GraphSAINT, GraphNas.

## References

Chiang et al. 2019
