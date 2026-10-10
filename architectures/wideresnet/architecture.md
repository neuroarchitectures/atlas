# Architecture: Wide ResNet

## Motivation

ResNet with wider channels (widen factor k) and fewer layers for efficient GPU training.

## Core Idea

ResNet with wider channels (widen factor k) and fewer layers for efficient GPU training.

## Architecture

### Overview

![wideresnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 32, 32] |
| 2 | Conv 1 (3x3) | `conv2d` | outFeatures=16 |
| 3 | Wide ResBlock Stage 1 | `conv2d` | outFeatures=160 |
| 4 | Wide ResBlock Stage 2 | `conv2d` | outFeatures=320 |
| 5 | Wide ResBlock Stage 3 | `conv2d` | outFeatures=640 |
| 6 | Global Average Pool | `adaptiveavgpool2d` | outFeatures=640 |
| 7 | FC Layer | `linear` | outFeatures=10 |
| 8 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Conv 1 (3x3)** — conv2d layer. 2. **Wide ResBlock Stage 1** — conv2d layer. 2. **Wide ResBlock Stage 2** — conv2d layer. 2. **Wide ResBlock Stage 3** — conv2d layer. 2. **Global Average Pool** — adaptiveavgpool2d layer. 2. **FC Layer** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — ResNet with wider channels (widen factor k) and fewer layers for efficient GPU training.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
