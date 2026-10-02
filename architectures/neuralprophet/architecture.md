# Architecture: NeuralProphet

## Motivation

Classical forecasting (ARIMA, Prophet) lacks flexibility for nonlinear patterns. Pure neural models lack interpretability.

## Core Idea

Combine interpretable components (trend, seasonality, autoregression) with neural network modules, maintaining Prophet's decomposable structure while adding neural flexibility.

## Architecture

### Overview

![neuralprophet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape: [96] |
| 2 | AR Net | `linear` | lag: 24 |
| 3 | Trend | `linear` |  |
| 4 | Seasonality | `fourier` |  |
| 5 | Lagged Covariates | `linear` |  |
| 6 | Add | `add` |  |
| 7 | Forecast | `output` |  |

</details>

### Components

1. **Trend** — Piecewise linear or logistic trend with changepoints. 2. **Seasonality** — Fourier series for weekly/yearly patterns. 3. **AR-Net** — Neural autoregressive component. 4. **Lagged covariates** — Neural network for exogenous variables. 5. **Additive composition** — All components summed for final forecast.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Combine interpretable components (trend, seasonality, autoregression) with neural network modules, maintaining Prophet's
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Prophet. Successor: N-BEATS, Temporal Fusion Transformer.

## References

Ostner et al. 2021
