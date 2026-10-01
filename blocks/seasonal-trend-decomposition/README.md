# Seasonal-Trend Decomposition

## Design Philosophy

Decompose a time series into trend (smooth, low-frequency) and seasonal (periodic) components. This is a fundamental operation in time series analysis, made differentiable for neural networks.

## Functionality

Moving average with kernel size k extracts the trend. The seasonal component is the residual: seasonal = x - trend. The decomposition is applied before and after each attention/processing block.

## Used By

Autoformer | FEDformer | Time series transformers

## Features

- **Differentiable**: Can be integrated into neural networks.
- **Interpretable**: Separates trend and seasonality.
- **No parameters**: Just a moving average filter.

## Evolution

Predecessor: STL decomposition (statistical). Successor: Learned decomposition blocks.
