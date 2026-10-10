# Architecture: iTransformer

## Motivation

Standard Transformers for time series embed temporal steps as tokens, which entangles multiple variates and fails to capture cross-variate correlations. iTransformer inverts this: each variate becomes a token.

## Core Idea

Embed each variate (time series channel) as a token. Apply self-attention across variates. Feed-forward networks process each variate independently. This captures inter-variate correlations naturally.

## Architecture

### Overview

![itransformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input Series | `input` | shape: [96, 8] (T=96, C=8 variates) |
| 2 | Variate Embedding | `linear` | each variate -> 512-dim token |
| 3 | Self-Attention | `attention` | across variates (not time) |
| 4 | Feed-Forward | `linear` | per-variate processing |
| 5 | Forecast | `output` | project back to horizon |

</details>

### Components

1. **Variate embedding** — Each variate's time series (length T) is embedded into a single token via a linear layer, treating variates as tokens. 2. **Self-attention across variates** — Standard multi-head attention, but the sequence dimension is over variates, capturing cross-variate dependencies. 3. **Feed-forward network** — Processes each variate's embedding independently (same as standard Transformer FFN). 4. **Projection** — Maps each variate token back to the forecast horizon.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Embed each variate (time series channel) as a token. Apply self-attention across variates. Feed-forward networks process each variate independently. This captures inter-variate correlations naturally.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: PatchTST, Transformer-based forecasting. Successor: iTransformer variants.

## References

- Liu et al. 2024
