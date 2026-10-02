# Architecture: Graph U-Net

## Motivation

U-Net architecture for graphs with graph pooling (gPool) and unpooling (gUnpool) operations.

## Core Idea

U-Net architecture for graphs with graph pooling (gPool) and unpooling (gUnpool) operations.

## Architecture

### Overview

![graph-unet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Graph Features | `input` | shape=[100, 100] |
| 2 | Graph Encoder 1 | `gnn` | hiddenSize=64 |
| 3 | gPool 1 | `gnn` | hiddenSize=64 |
| 4 | Graph Encoder 2 | `gnn` | hiddenSize=128 |
| 5 | gPool 2 | `gnn` | hiddenSize=128 |
| 6 | Bottleneck | `gnn` | hiddenSize=256 |
| 7 | gUnpool 2 | `gnn` | hiddenSize=128 |
| 8 | Graph Decoder 2 | `gnn` | hiddenSize=128 |
| 9 | gUnpool 1 | `gnn` | hiddenSize=64 |
| 10 | Node Predictions | `output` | — |

</details>

### Components

1. **Graph Features** — input layer. 2. **Graph Encoder 1** — gnn layer. 2. **gPool 1** — gnn layer. 2. **Graph Encoder 2** — gnn layer. 2. **gPool 2** — gnn layer. 2. **Bottleneck** — gnn layer. 2. **gUnpool 2** — gnn layer. 2. **Graph Decoder 2** — gnn layer. 2. **gUnpool 1** — gnn layer. 2. **Node Predictions** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — U-Net architecture for graphs with graph pooling (gPool) and unpooling (gUnpool) operations.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
