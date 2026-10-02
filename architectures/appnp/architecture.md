# Architecture: APPNP

## Motivation

GCNs entangle feature transformation and propagation, causing over-smoothing in deep networks. APPNP decouples them: an MLP transforms features, then Personalized PageRank propagates predictions.

## Core Idea

A neural network (MLP) first transforms node features into predictions. Then Personalized PageRank (PPR) propagates these predictions across the graph with teleport probability alpha (keeping original predictions). This decouples feature learning from graph structure.

## Architecture

### Overview

![appnp architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | N x F |
| 2 | MLP Predictor | `linear` | F -> num_classes |
| 3 | Personalized PageRank | `identity` | K=10 hops, alpha=0.1 |
| 4 | Node Classification | `output` | per-node labels |

</details>

### Components

1. **MLP predictor** — transforms features to predictions independently of graph. 2. **Personalized PageRank** — Z = alpha*(I - (1-alpha)A)^{-1} * H. 3. **Teleport probability** — alpha controls how much original prediction is retained. 4. **No over-smoothing** — PPR propagation preserves node identity. 5. **Decoupled design** — feature learning and propagation are separate.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A neural network (MLP) first transforms node features into predictions. Then Personalized PageRank (PPR) propagates thes...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GCN (entangled). Successor: GPR-GNN (learned propagation weights).

## References

- Klicpera et al. 2019
