# Architecture: WaveNet

## Motivation

Generating raw audio waveforms (24000 samples/sec) autoregressively with RNNs is computationally prohibitive. WaveNet uses dilated causal convolutions to achieve large receptive fields efficiently.

## Core Idea

Stack dilated causal convolutions with exponentially increasing dilation rates (1, 2, 4, 8, ..., 512). Use gated activation: tanh(W_f * x) * sigmoid(W_g * x). Residual and skip connections aggregate multi-scale features.

## Architecture

### Overview

![wavenet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Audio Sample | `input` | shape: [1] |
| 2 | Causal Conv (d=1) | `conv` | kernelSize: 2, dilation: 1 |
| 3 | Gated Activation | `multiply` | tanh * sigmoid |
| 4 | Residual | `add` |  |
| 5 | Causal Conv (d=2) | `conv` | kernelSize: 2, dilation: 2 |
| 6 | Gated Activation 2 | `multiply` |  |
| 7 | Skip Connections | `identity` | summed across all layers |
| 8 | Next Sample | `output` | softmax over 256 quantization levels |

</details>

### Components

1. **Dilated causal convolutions** — Causal (only past context) with exponentially growing dilation (1, 2, 4, ..., 512), enabling a receptive field of thousands of samples efficiently. 2. **Gated activation** — tanh(W_f * x) * sigmoid(W_g * x), more expressive than ReLU for audio generation. 3. **Residual and skip connections** — Residual connections within each dilation block; skip connections aggregate outputs from all layers. 4. **Conditional generation** — Global conditioning (speaker identity) via learned embeddings, local conditioning (linguistic features) via transposed convolutions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Stack dilated causal convolutions with exponentially increasing dilation rates (1, 2, 4, 8, ..., 512). Use gated activation: tanh(W_f * x) * sigmoid(W_g * x). Residual and skip connections aggregate m
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SampleRNN. Successor: Parallel WaveNet, WaveGlow, HiFi-GAN.

## References

- van den Oord et al. 2016
