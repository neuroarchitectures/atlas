# Architecture: FEDformer

## Motivation

Standard Transformers for time series are O(n²) and struggle with long horizons. Most time series energy is in low-frequency components.

## Core Idea

Use Fourier and wavelet transforms to process time series in the frequency domain with linear complexity, capturing dominant seasonal patterns while discarding high-frequency noise.

## Architecture

### Overview

![fedformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape: [96, 7] |
| 2 | Decomposition | `series-decomp` |  |
| 3 | Fourier Attention | `fft` | numModes: 64 |
| 4 | Frequency Linear | `linear` |  |
| 5 | Final Decomposition | `series-decomp` |  |
| 6 | Forecast | `output` |  |

</details>

### Components

1. **Trend-seasonal decomposition** — Moving average separates trend from seasonal. 2. **Fourier attention** — Sparse attention in frequency domain using top-k modes. 3. **Frequency-domain MLP** — Linear projection in Fourier space. 4. **Inversion** — IFFT to produce time-domain forecast.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use Fourier and wavelet transforms to process time series in the frequency domain with linear complexity, capturing domi
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Autoformer, Informer. Successor: TimesNet, iTransformer.

## References

Zhou et al. 2022
