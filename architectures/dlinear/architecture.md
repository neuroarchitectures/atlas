# Architecture: DLinear

## Motivation

Complex Transformer-based time series models may overfit and are computationally expensive. Simple linear models may suffice.

## Core Idea

Decompose the series into trend and seasonal components using moving average, then apply a simple linear layer to each component independently.

## Architecture

### Overview

![dlinear architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape: [96] |
| 2 | Decomposition | `series-decomp` |  |
| 3 | Trend Linear | `linear` |  |
| 4 | Seasonal Linear | `linear` |  |
| 5 | Add | `add` |  |
| 6 | Forecast | `output` |  |

</details>

### Components

1. **Series decomposition** — Moving average extracts trend; residual is seasonal. 2. **Trend linear** — Single linear layer mapping input to output horizon. 3. **Seasonal linear** — Separate linear layer for seasonal component. 4. **Addition** — Sum trend and seasonal forecasts.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Decompose the series into trend and seasonal components using moving average, then apply a simple linear layer to each c
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: N-BEATS, Autoformer. Successor: PatchMixer, TimeMixer.

## References

Zeng et al. 2023
