# Architecture: Autoformer

## Motivation

Standard attention in Transformers for time series is point-wise and doesn't capture periodic patterns. Autoformer replaces attention with auto-correlation based on the Fast Fourier Transform.

## Core Idea

Replace self-attention with an auto-correlation mechanism that discovers period-based dependencies using FFT, and use a series decomposition block that separates trend and seasonal components.

## Architecture

### Overview

![autoformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input Series | `input` | shape: [96, 8] |
| 2 | Series Decomposition | `identity` | moving avg trend/seasonal split |
| 3 | Auto-Correlation | `attention` | FFT-based period discovery |
| 4 | Series Decomp 2 | `identity` | decompose after attention |
| 5 | Trend Accumulation | `linear` | accumulate trend across layers |
| 6 | Forecast | `output` | seasonal + trend |

</details>

### Components

1. **Series decomposition block** — Uses a moving average to separate the input into trend (smooth) and seasonal (residual) components. 2. **Auto-correlation mechanism** — Replaces self-attention. Uses FFT to find top-k periods in the signal, then aggregates time-delayed features at those periods. O(L log L) complexity. 3. **Accumulation** — Trend components are accumulated across layers; seasonal components flow through auto-correlation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace self-attention with an auto-correlation mechanism that discovers period-based dependencies using FFT, and use a series decomposition block that separates trend and seasonal components.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Informer. Successor: FEDformer, iTransformer.

## References

- Wu et al. 2021
