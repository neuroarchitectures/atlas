# Architecture: GATv2

## Motivation

GAT uses static attention: the attention function is a linear function of node features, limiting its expressivity.

## Core Idea

Apply a shared linear transformation after the attention nonlinearity, making the attention function dynamic and more expressive.

## Architecture

### Overview

![gatv2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | shape: [1433] |
| 2 | Linear Projection | `linear` | outFeatures: 8 |
| 3 | GATv2 Layer | `gat-layer` | heads: 8 |
| 4 | GATv2 Layer | `gat-layer` | heads: 1 |
| 5 | Classifier | `linear` | outFeatures: 7 |
| 6 | Node Classification | `output` |  |

</details>

### Components

1. **Linear projection** — Projects node features to lower dimension. 2. **GATv2 layers** — Attention with W*a + W*b instead of a^T W b, enabling dynamic attention. 3. **Multi-head attention** — Multiple attention heads concatenated. 4. **Classifier** — Linear layer for node classification.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Apply a shared linear transformation after the attention nonlinearity, making the attention function dynamic and more ex
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GAT. Successor: GATv3, Graphormer.

## References

Brody et al. 2022
