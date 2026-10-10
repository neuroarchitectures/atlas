# Architecture: DeepAR

## Motivation

Probabilistic forecasting with autoregressive recurrent network producing distribution parameters.

## Core Idea

Probabilistic forecasting with autoregressive recurrent network producing distribution parameters.

## Architecture

### Overview

![deepar architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series Window | `input` | shape=[168, 1] |
| 2 | LSTM Encoder (3 layers) | `lstm` | hiddenSize=40, numLayers=3 |
| 3 | Distribution Parameters | `linear` | outFeatures=2 |
| 4 | Forecast Distribution | `output` | — |

</details>

### Components

1. **Time Series Window** — input layer. 2. **LSTM Encoder (3 layers)** — lstm layer. 2. **Distribution Parameters** — linear layer. 2. **Forecast Distribution** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Probabilistic forecasting with autoregressive recurrent network producing distribution parameters.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
