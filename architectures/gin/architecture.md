# Architecture: GIN

## Motivation

GNNs using mean/max aggregation cannot distinguish certain non-isomorphic graphs that the Weisfeiler-Lehman (WL) test can. GIN addresses this by using sum aggregation, which is injective on multisets and therefore maximally powerful among message-passing GNNs.

## Core Idea

Use sum aggregation (injective on multisets) combined with an MLP to update node representations. This makes GIN as powerful as the 1-WL test, which is the theoretical upper bound for message-passing GNNs.

## Architecture

### Overview

![gin architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | shape: [2708, 1433] |
| 2 | Sum Aggregation | `custom` | type: neighborhood_sum |
| 3 | Linear (1+ε) | `linear` | outFeatures: 64, inFeatures: 1433 |
| 4 | ReLU | `relu` |  |
| 5 | Linear 2 | `linear` | outFeatures: 64, inFeatures: 64 |
| 6 | Sum Aggregation | `custom` | type: neighborhood_sum |
| 7 | Linear (1+ε) | `linear` | outFeatures: 64, inFeatures: 64 |
| 8 | ReLU | `relu` |  |
| 9 | Linear 2 | `linear` | outFeatures: 7, inFeatures: 64 |
| 10 | Output | `output` |  |

</details>

Use sum aggregation (injective on multisets) combined with an MLP to update node representations. This makes GIN as powerful as the 1-WL test, which is the theoretical upper bound for message-passing GNNs.

### Components

2. **Sum Aggregation** (`custom`, scope: `layer.0`) — Params: type: neighborhood_sum
3. **Linear (1+ε)** (`linear`, scope: `layer.0`) — Params: outFeatures: 64, inFeatures: 1433
4. **ReLU** (`relu`, scope: `layer.0`) — Params: none
5. **Linear 2** (`linear`, scope: `layer.0`) — Params: outFeatures: 64, inFeatures: 64
6. **Sum Aggregation** (`custom`, scope: `layer.1`) — Params: type: neighborhood_sum
7. **Linear (1+ε)** (`linear`, scope: `layer.1`) — Params: outFeatures: 64, inFeatures: 64
8. **ReLU** (`relu`, scope: `layer.1`) — Params: none
9. **Linear 2** (`linear`, scope: `layer.1`) — Params: outFeatures: 7, inFeatures: 64

### Data Flow

The architecture processes input through a sequence of 10 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to GIN.

## Evolution

GIN builds on GCN and GraphSAGE but provides a theoretical guarantee of maximum expressiveness among message-passing GNNs. It is as powerful as the 1-WL test. Successors include GIN-AK (adaptive kernels), higher-order GNNs (k-GNN), and graph transformers.

## Source

- **Paper:** arXiv:1810.00826
- **Year:** 2019
- **Authors:** Xu et al.
