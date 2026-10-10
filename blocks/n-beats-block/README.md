# N-BEATS Block

## Design Philosophy

A fully-connected block that produces both a forecast (future prediction) and a backcast (reconstruction of the input). The backcast is subtracted from the input, forming a doubly residual structure.

## Functionality

1. Shared MLP (4 layers, ReLU). 2. Two linear heads: backcast (same shape as input) and forecast (horizon). 3. Backcast is subtracted from input (backward residual). 4. Forecast is accumulated (forward residual).

## Used By

N-BEATS | N-HiTS | Time series forecasting

## Features

- **Doubly residual**: Backward (input - backcast) and forward (sum of forecasts).
- **Interpretable**: Can be constrained to produce trend/seasonality basis functions.
- **Generic**: No time-series-specific features; pure MLP.

## Evolution

Predecessor: DeepAR. Successor: N-HiTS (multi-rate sampling), N-BEATS-S.
