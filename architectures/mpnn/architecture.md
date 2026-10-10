# Architecture: MPNN

## Motivation

Many GNN variants (GGNN, Neural Fingerprint, interaction nets, etc.) share the same structure but use different message/aggregation/update functions. MPNN formalizes this common abstraction into a single framework, enabling systematic comparison and transfer of ideas.

## Core Idea

Define a general message-passing framework: (1) compute messages m = M(h_v, h_w, e_vw) for each edge, (2) aggregate messages via sum/mean, (3) update node states h_v' = U(h_v, aggregated_m). This unifies GCN, GAT, GraphSAGE, GGNN, and others under a common formalism.

## Architecture

### Overview

![mpnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | shape: [100, 64] |
| 2 | Edge Features | `input` | shape: [500, 16] |
| 3 | Message Function M | `custom` | type: mlp_message, hiddenDim: 128 |
| 4 | Edge Transform | `custom` | type: edge_mlp, hiddenDim: 128 |
| 5 | Aggregation (Sum) | `custom` | type: sum_aggregation |
| 6 | Update Function U | `custom` | type: gru_update, hiddenDim: 64 |
| 7 | Readout R | `custom` | type: set_transformer_readout, hiddenDim: 64 |
| 8 | Output | `linear` | outFeatures: 10, inFeatures: 64 |
| 9 | Graph Output | `output` |  |

</details>

Define a general message-passing framework: (1) compute messages m = M(h_v, h_w, e_vw) for each edge, (2) aggregate messages via sum/mean, (3) update node states h_v' = U(h_v, aggregated_m). This unifies GCN, GAT, GraphSAGE, GGNN, and others under a common formalism.

### Components

3. **Message Function M** (`custom`, scope: `layer.0`) — Params: type: mlp_message, hiddenDim: 128
4. **Edge Transform** (`custom`, scope: `layer.0`) — Params: type: edge_mlp, hiddenDim: 128
5. **Aggregation (Sum)** (`custom`, scope: `layer.0`) — Params: type: sum_aggregation
6. **Update Function U** (`custom`, scope: `layer.0`) — Params: type: gru_update, hiddenDim: 64
7. **Readout R** (`custom`, scope: `model`) — Params: type: set_transformer_readout, hiddenDim: 64
8. **Output** (`linear`, scope: `model`) — Params: outFeatures: 10, inFeatures: 64

### Data Flow

The architecture processes input through a sequence of 9 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to MPNN.

## Evolution

MPNN is a generalization framework that unifies GCN, GAT, GraphSAGE, GGNN, and others. It is the basis for most molecular GNN architectures (DMPNN, ChemProp, etc.). Successors include Graph Transformers and higher-order message passing schemes.

## Source

- **Paper:** arXiv:1704.01212
- **Year:** 2017
- **Authors:** Gilmer et al.
