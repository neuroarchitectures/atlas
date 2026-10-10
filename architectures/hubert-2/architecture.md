# Architecture: HuBERT-2

## Motivation

Self-supervised speech models need stable training targets. Contrastive methods are sensitive to representation collapse.

## Core Idea

Use masked prediction of clustered acoustic features (k-means on MFCC or hidden states), iteratively refining the targets using the model's own representations.

## Architecture

### Overview

![hubert-2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Waveform | `input` | shape: [16000] |
| 2 | Conv1d | `conv1d` |  |
| 3 | Conv1d | `conv1d` |  |
| 4 | Conv1d | `conv1d` |  |
| 5 | Conv1d | `conv1d` |  |
| 6 | Conv1d | `conv1d` |  |
| 7 | Projection | `conv1d` |  |
| 8 | Transformer | `transformer` | 24 layers |
| 9 | Hidden States | `output` |  |

</details>

### Components

1. **Convolutional feature extractor** — 7-layer 1D CNN that converts raw waveform to 50Hz feature sequence. 2. **Transformer encoder** — 24-layer transformer with relative positional encoding. 3. **Masked prediction** — Randomly mask ~8% of frames, predict cluster assignments. 4. **Iterative refinement** — Use first-pass hidden states as new k-means targets for second pass.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use masked prediction of clustered acoustic features (k-means on MFCC or hidden states), iteratively refining the target
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: wav2vec 2.0, HuBERT. Successor: WavLM, data2vec.

## References

Hsu et al. 2021
