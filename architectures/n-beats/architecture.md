# Architecture: N-BEATS

## Motivation

Time series forecasting was dominated by statistical methods. N-BEATS demonstrates that a pure deep learning approach with no domain-specific feature engineering can outperform statistical baselines.

## Core Idea

A stack of blocks, each with shared MLP layers. Each block produces a forecast and a backcast. The backcast is subtracted from the input (residual), and the forecast is accumulated. This decomposition enables interpretable trend and seasonality extraction.

## Architecture

### Overview

![n-beats architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Lookback Window | `input` | shape: [168] (e.g., 7 days hourly) |
| 2 | Block 1 FC | `linear` | outFeatures: 256, inFeatures: 168 |
| 3 | ReLU | `relu` |  |
| 4 | Block 1 FC 2 | `linear` | outFeatures: 256, inFeatures: 256 |
| 5 | Backcast | `linear` | outFeatures: 168 (same as input) |
| 5 | Forecast | `linear` | outFeatures: 24 (horizon) |
| 6 | Residual | `output` | input - backcast for next block |
| 6 | Forecast | `output` | accumulated forecast |

</details>

### Components

1. **Stack of blocks** — Each block has shared MLP layers (4 layers, 256 units). 2. **Backcast** — Linear projection producing a prediction of the input (same shape). Subtracted from input to form residual. 3. **Forecast** — Linear projection producing the horizon forecast. Accumulated across blocks. 4. **Doubly residual stacking** — Backward residual (input - backcast) feeds next block; forward residual (sum of forecasts) accumulates prediction. 5. **Interpretability** — By constraining the final layer of certain blocks to produce basis functions (trend: polynomial, seasonality: Fourier), the model becomes interpretable.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A stack of blocks, each with shared MLP layers. Each block produces a forecast and a backcast. The backcast is subtracted from the input (residual), and the forecast is accumulated. This decomposition
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DeepAR, statistical methods. Successor: N-HiTS, N-BEATS-S, N-BEATS-I.

## References

- Oreshkin et al. 2020
