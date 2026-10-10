# Architecture: N-HiTS

## Motivation

N-BEATS processes the full lookback window at every block, creating O(L^2) complexity. N-HiTS reduces this by downsampling the input at different rates for different blocks.

## Core Idea

Each block receives a different pooling rate of the input (1x, 2x, 4x, etc.), enabling multi-scale analysis. Combined with hierarchical interpolation for the forecast output, this reduces complexity from O(L^2) to O(L).

## Architecture

### Overview

![n-hits architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Lookback Window | `input` | shape: [512] |
| 2 | MaxPool (rate=1) | `identity` | multi-rate downsampling |
| 3 | Block 1 FC | `linear` | outFeatures: 256, inFeatures: 512 |
| 4 | ReLU | `relu` |  |
| 5 | Block 1 FC 2 | `linear` | outFeatures: 256, inFeatures: 256 |
| 6 | Backcast | `linear` | outFeatures: 512 |
| 6 | Forecast + Interp | `linear` | hierarchical interpolation |
| 7 | Residual | `output` | input - backcast |
| 7 | Forecast | `output` | accumulated |

</details>

### Components

1. **Multi-rate data sampling** — Each block receives input pooled at a different rate (1, 2, 4, ...), enabling multi-scale temporal pattern extraction. 2. **Hierarchical interpolation** — The forecast head produces a coarse forecast that is interpolated to the full horizon, reducing parameters. 3. **Doubly residual stacking** — Same as N-BEATS: backward residual feeds next block, forward residual accumulates forecast.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Each block receives a different pooling rate of the input (1x, 2x, 4x, etc.), enabling multi-scale analysis. Combined with hierarchical interpolation for the forecast output, this reduces complexity f
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: N-BEATS. Successor: N-HiTS variants for hierarchical forecasting.

## References

- Challu et al. 2023
