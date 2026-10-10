# Architecture: Temporal Fusion Transformer

## Motivation

Real-world forecasting requires handling static metadata, known future inputs, and past observations simultaneously. TFT combines all these with interpretable attention and variable selection.

## Core Idea

Use a multi-modal architecture: LSTM for local temporal processing, multi-head attention for long-range dependencies, variable selection networks for feature importance, and gated residual networks for flexible non-linear processing.

## Architecture

### Overview

![temporal fusion transformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Past Observations | `input` | shape: [168, 10] |
| 2 | Variable Selection | `attention` | static + time-varying input selection |
| 3 | LSTM Encoder | `lstm` | hiddenSize: 128, local temporal processing |
| 4 | Interpretable Attention | `attention` | multi-head attention for long-range |
| 5 | Gated Residual | `linear` | GRN with skip connections |
| 6 | Quantile Forecast | `output` | quantile predictions (P10, P50, P90) |

</details>

### Components

1. **Variable Selection Network (VSN)** — Learns which input variables are relevant, providing feature importance. 2. **LSTM encoder-decoder** — Processes local temporal patterns. 3. **Interpretable multi-head attention** — Single-head attention per head, enabling per-head temporal pattern interpretation. 4. **Gated Residual Network (GRN)** — ELU + GLU + skip connection, enables flexible non-linear processing or identity. 5. **Quantile output** — Produces P10, P50, P90 quantiles for probabilistic forecasting.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a multi-modal architecture: LSTM for local temporal processing, multi-head attention for long-range dependencies, variable selection networks for feature importance, and gated residual networks fo
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DeepAR, Transformer-based forecasting. Successor: TFT variants for hierarchical/multi-target forecasting.

## References

- Lim et al. 2021
