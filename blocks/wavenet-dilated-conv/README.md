# WaveNet Dilated Causal Convolution

## Design Philosophy

Stack dilated causal convolutions with exponentially growing dilation rates (1, 2, 4, 8, ..., 512). This creates a large receptive field (thousands of samples) while keeping the number of parameters small.

## Functionality

Causal conv (only past context) with dilation d: output[i] = sum_j W[j] * input[i - d*j]. Stack layers with d = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 for a receptive field of 1024.

## Used By

WaveNet | SampleRNN | Temporal Convolutional Networks (TCN)

## Features

- **Large receptive field**: Exponential growth with depth.
- **Causal**: Only uses past context (autoregressive).
- **Efficient**: Few parameters for large context.

## Evolution

Predecessor: Causal conv. Successor: Parallel WaveNet, HiFi-GAN (non-autoregressive).
