# Architecture: Enformer

## Motivation

Predicting gene expression from DNA requires modeling very long-range interactions (up to 100kb) that exceed the receptive field of CNNs.

## Core Idea

Replace the CNN trunk with a Transformer after convolutional downsampling, enabling attention over ~100kb context windows.

## Architecture

### Overview

![enformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | DNA Sequence | `input` | shape: [196608, 4] |
| 2 | Conv Block | `conv1d` | outChannels: 1536 |
| 3 | Pooling | `maxpool1d` |  |
| 4 | Conv Block | `conv1d` | outChannels: 3072 |
| 5 | Pooling | `maxpool1d` |  |
| 6 | Transformer | `transformer` | 11 blocks |
| 7 | Output Head | `linear` | outFeatures: 164 |
| 8 | Gene Expression | `output` |  |

</details>

### Components

1. **Convolutional trunk** — Seven Conv1d blocks with downsampling reduce sequence length while expanding channels. 2. **Transformer trunk** — 11 Transformer blocks with relative positional encoding capture long-range genomic interactions. 3. **Output head** — Predicts 5313 human and 164 mouse tracks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace the CNN trunk with a Transformer after convolutional downsampling, enabling attention over ~100kb context window
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Basenji, DeepSEA. Successor: Borzoi, Sei.

## References

Avsec et al. 2021
