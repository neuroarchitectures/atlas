# Architecture: GraphMAE

## Motivation

Masked autoencoder for graphs: reconstruct masked node features via GNN encoder-decoder.

## Core Idea

Masked autoencoder for graphs: reconstruct masked node features via GNN encoder-decoder.

## Architecture

### Overview

![graphmae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Graph Features | `input` | shape=[100, 500] |
| 2 | GNN Encoder | `gnn` | hiddenSize=256 |
| 3 | GNN Decoder (reconstruction) | `gnn` | hiddenSize=500 |
| 4 | Reconstructed Features | `output` | — |

</details>

### Components

1. **Graph Features** — input layer. 2. **GNN Encoder** — gnn layer. 2. **GNN Decoder (reconstruction)** — gnn layer. 2. **Reconstructed Features** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Masked autoencoder for graphs: reconstruct masked node features via GNN encoder-decoder.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
