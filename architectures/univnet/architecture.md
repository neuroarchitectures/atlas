# Architecture: UnivNet

## Motivation

Neural vocoders must produce high-fidelity waveforms from spectral features. Fixed filter kernels limit adaptability.

## Core Idea

Use a kernel predictor network that generates convolutional filter kernels conditioned on the input spectrogram, enabling input-adaptive filtering.

## Architecture

### Overview

![univnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Mel Spectrogram | `input` | shape: [80] |
| 2 | Kernel Predictor | `conv2d` |  |
| 3 | LSTM | `lstm` | hiddenSize: 64 |
| 4 | Neural Filter | `conv1d` |  |
| 5 | Neural Filter | `conv1d` |  |
| 6 | Waveform | `output` |  |

</details>

### Components

1. **Kernel predictor** — LSTM + Conv network that generates filter kernels from mel-spectrogram. 2. **Neural filter block** — Multiple transposed convolutions with predicted kernels. 3. **Multi-resolution STFT discriminator** — Adversarial training with spectrograms at multiple resolutions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a kernel predictor network that generates convolutional filter kernels conditioned on the input spectrogram, enablin
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: HiFi-GAN, WaveNet. Successor: BigVGAN.

## References

Jang et al. 2021
