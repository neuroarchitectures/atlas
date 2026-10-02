# Architecture: CoAtNet

## Motivation

Hybrid CNN-Transformer: convolution stages with attention stages, combining inductive biases.

## Core Idea

Hybrid CNN-Transformer: convolution stages with attention stages, combining inductive biases.

## Architecture

### Overview

![coatnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 224, 224] |
| 2 | Conv Stage 1 (MBConv) | `conv2d` | outFeatures=64 |
| 3 | Conv Stage 2 (MBConv) | `conv2d` | outFeatures=96 |
| 4 | Transformer Stage 1 | `transformer-encoder` | numLayers=2, hiddenSize=192, numHeads=3 |
| 5 | Transformer Stage 2 | `transformer-encoder` | numLayers=12, hiddenSize=384, numHeads=6 |
| 6 | Classification Head | `linear` | outFeatures=1000 |
| 7 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Conv Stage 1 (MBConv)** — conv2d layer. 2. **Conv Stage 2 (MBConv)** — conv2d layer. 2. **Transformer Stage 1** — transformer-encoder layer. 2. **Transformer Stage 2** — transformer-encoder layer. 2. **Classification Head** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Hybrid CNN-Transformer: convolution stages with attention stages, combining inductive biases.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
