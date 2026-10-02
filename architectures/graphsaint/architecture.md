# Architecture: GraphSAINT

## Motivation

Cluster-GCN introduces bias by partitioning the graph. Full-batch GCN is memory-limited. A sampling approach with variance reduction is needed.

## Core Idea

Sample subgraphs (nodes, edges, or random walks) before each training step, with normalization to ensure unbiased gradient estimates.

## Architecture

### Overview

![graphsaint architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | shape: [2600] |
| 2 | GCN Layer 1 | `gcn-layer` | outChannels: 256 |
| 3 | ReLU | `relu` |  |
| 4 | GCN Layer 2 | `gcn-layer` | outChannels: 10 |
| 5 | Node Classification | `output` |  |

</details>

### Components

1. **Subgraph sampler** — Samples a small subgraph per batch (node, edge, or walk-based). 2. **Normalization** — Variance reduction to ensure unbiased estimates. 3. **GCN layers** — Standard graph convolution on sampled subgraph. 4. **Aggregation** — Per-node predictions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Sample subgraphs (nodes, edges, or random walks) before each training step, with normalization to ensure unbiased gradie
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GraphSAGE, Cluster-GCN. Successor: DeeperGCN, GraphNas.

## References

Zeng et al. 2020
