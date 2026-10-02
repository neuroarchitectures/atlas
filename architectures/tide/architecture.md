# Architecture: TiDE

## Motivation

Transformer-based time series models are computationally expensive for long horizons. TiDE shows that a simple MLP with residual connections can match or exceed transformer performance while being 10x faster.

## Core Idea

Past values and covariates are flattened and passed through a deep MLP encoder. Multiple dense layers with residual connections process the representation. A dense decoder projects to the forecast horizon. Temporal decoder captures residual temporal patterns.

## Architecture

### Overview

![tide architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Past + Covariates | `input` | lookback x features |
| 2 | Dense Encoder | `linear` | flatten -> 2048 |
| 3 | Hidden Dense Layers | `linear` | residual MLP blocks |
| 4 | Dense Decoder | `linear` | 2048 -> horizon x features |
| 5 | Forecast | `output` | future values |

</details>

### Components

1. **Flatten input** — past values + covariates flattened to vector. 2. **Dense encoder** — MLP with residual connections. 3. **Feature projection** — projects covariates. 4. **Dense decoder** — MLP projects to forecast horizon. 5. **Temporal decoder** — residual MLP captures temporal patterns. 6. **10x faster** — no attention, pure MLP.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Past values and covariates are flattened and passed through a deep MLP encoder. Multiple dense layers with residual conn...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: N-BEATS, MLP baseline. Successor: TimeMixer, modern MLP-based forecasters.

## References

- Das et al. 2023
