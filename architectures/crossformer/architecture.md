# Architecture: Crossformer

## Motivation

Time series transformers typically attend only across time. Multivariate series have both temporal and cross-dimension dependencies. Crossformer uses dual attention to capture both.

## Core Idea

Input is segmented into patches. A two-stage embedding captures both individual and group patterns. The encoder uses cross-time and cross-dimension attention alternately. A decoder produces the forecast.

## Architecture

### Overview

![crossformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Multivariate Time Series | `input` | T x D |
| 2 | Segment Patching | `identity` | divide into patches |
| 3 | Two-Stage Embedding | `linear` | individual + group |
| 4 | Transformer Encoder | `linear` | cross-time + cross-dim attention |
| 5 | Decoder | `linear` | linear projection |
| 6 | Forecast | `output` | future values |

</details>

### Components

1. **Segment patching** — divides time series into segments for efficiency. 2. **Two-stage embedding** — stage 1 embeds individual patches, stage 2 embeds groups. 3. **Cross-time attention** — attends across time steps. 4. **Cross-dimension attention** — attends across variables/dimensions. 5. **Dual-scale attention** — captures both fine and coarse patterns.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Input is segmented into patches. A two-stage embedding captures both individual and group patterns. The encoder uses cro...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Informer, Autoformer. Successor: iTransformer, Patch-TST.

## References

- Zhang & Yan 2023
