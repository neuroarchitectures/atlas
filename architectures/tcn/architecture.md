# Architecture: Temporal Convolutional Network

## Motivation

RNNs process sequences sequentially, preventing parallelization. TCN uses dilated causal convolutions to achieve large receptive fields while allowing parallel computation and stable gradients.

## Core Idea

Stacked 1D convolution blocks with exponentially increasing dilation rates (1, 2, 4, 8, ...) cover long-range dependencies. Causal padding ensures no future leakage. Residual connections stabilize deep stacks.

## Architecture

### Overview

![tcn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | variable length |
| 2 | Dilated Conv Block 1 | `conv1d` | dilation=1, 64 filters |
| 3 | Dilated Conv Block 2 | `conv1d` | dilation=2, 64 filters |
| 4 | Dilated Conv Block 3 | `conv1d` | dilation=4, 64 filters |
| 5 | Output Layer | `linear` | 64 -> prediction |
| 6 | Prediction | `output` | next-step forecast |

</details>

### Components

1. **Causal convolutions** — zero-padding on the left ensures no future information leaks. 2. **Dilation** — exponentially growing dilation (1, 2, 4, 8...) for large receptive field. 3. **Residual blocks** — each block has two conv layers + skip connection. 4. **Parallel computation** — all timesteps processed simultaneously. 5. **Stable gradients** —no vanishing gradient problem unlike RNNs.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Stacked 1D convolution blocks with exponentially increasing dilation rates (1, 2, 4, 8, ...) cover long-range dependenci...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: WaveNet (audio), RNN/LSTM. Successor: Patch-TST, iTransformer (transformer-based).

## References

- Bai et al. 2018
