# Architecture: Pyraformer

## Motivation

Pyramidal attention with multi-resolution structure for long-range time series forecasting.

## Core Idea

Pyramidal attention with multi-resolution structure for long-range time series forecasting.

## Architecture

### Overview

![pyraformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape=[720, 1] |
| 2 | Embedding | `embedding` | vocabSize=1000, dim=512 |
| 3 | Pyramidal Attention | `transformer-encoder` | numLayers=4, hiddenSize=512, numHeads=8 |
| 4 | Forecast | `output` | — |

</details>

### Components

1. **Time Series** — input layer. 2. **Embedding** — embedding layer. 2. **Pyramidal Attention** — transformer-encoder layer. 2. **Forecast** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Pyramidal attention with multi-resolution structure for long-range time series forecasting.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
