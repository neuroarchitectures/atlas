# Architecture: SCINet

## Motivation

Standard time series models process data sequentially and miss multi-resolution temporal patterns.

## Core Idea

Use down-sampling to split the series into even/odd subsequences at multiple levels, then apply convolutional interactions between them to capture temporal patterns at different resolutions.

## Architecture

### Overview

![scinet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape: [96, 7] |
| 2 | Down-sample | `downsample` |  |
| 3 | Down-sample | `downsample` |  |
| 4 | Conv | `conv1d` |  |
| 5 | Conv | `conv1d` |  |
| 6 | Concat | `concat` |  |
| 7 | Output | `linear` |  |
| 8 | Forecast | `output` |  |

</details>

### Components

1. **Down-sampling** — Split series into even and odd indexed subsequences. 2. **Convolutional interaction** — 1D convolutions extract local patterns. 3. **Multi-level stacking** — Multiple levels of down-sampling and interaction. 4. **Output projection** — Linear layer maps to forecast horizon.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use down-sampling to split the series into even/odd subsequences at multiple levels, then apply convolutional interactio
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: N-BEATS, N-HiTS. Successor: TimesNet.

## References

Liu et al. 2022
