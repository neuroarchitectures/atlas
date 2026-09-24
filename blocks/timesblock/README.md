# TimesBlock (1D-to-2D Time-Series Block)

## Design Philosophy

Transform 1D time series into a set of **2D tensors** based on multiple periods (discovered via FFT), embedding intraperiod-variations into columns and interperiod-variations into rows. The philosophy: time series have multi-periodicity; analyzing 1D is hard, but a 2D representation lets a single 2D conv capture both within-period and between-period variations simultaneously.

## Functionality

1. **Period discovery (FFT)**: Apply FFT to the 1D series; select top-k dominant frequencies as periods.
2. **1D-to-2D reshape**: For each period `p_i`, reshape the 1D series into a 2D tensor `[p_i, length/p_i]` — columns = within-period, rows = between-period.
3. **2D inception block**: Multi-scale 2D conv captures variations at different granularities.
4. **2D-to-1D reshape**: Convert back to 1D.
5. **Aggregation**: Combine results from different periods (adaptive weighting).
- Stacked N TimesBlocks form the full network.

## Used By

| Model | Role |
|-------|------|
| TimesNet | The defining backbone (forecasting, imputation, classification, anomaly detection) |

## Features

- **Multi-periodicity**: FFT discovers periods data-driven (no manual specification).
- **2D analysis**: A single 2D kernel captures intra- and inter-period variations.
- **Task-general**: Same backbone for forecasting, imputation, classification, anomaly detection.

## Evolution

- **Predecessor**: RNN/TCN/Transformer-based time-series models; Inception (2D conv).
- **Successor**: PatchTST (patching + channel independence); iTransformer (inverted transformer); TimeMixer (multi-scale MLP).
