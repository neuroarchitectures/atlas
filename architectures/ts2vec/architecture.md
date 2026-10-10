# Architecture: TS2Vec

## Motivation

Universal contrastive representation learning for time series with hierarchical contextual masking.

## Core Idea

Universal contrastive representation learning for time series with hierarchical contextual masking.

## Architecture

### Overview

![ts2vec architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape=[168, 1] |
| 2 | Dilated Conv Encoder | `conv1d` | outFeatures=512 |
| 3 | Projection Head | `linear` | outFeatures=320 |
| 4 | Time Series Representation | `output` | — |

</details>

### Components

1. **Time Series** — input layer. 2. **Dilated Conv Encoder** — conv1d layer. 2. **Projection Head** — linear layer. 2. **Time Series Representation** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Universal contrastive representation learning for time series with hierarchical contextual masking.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
