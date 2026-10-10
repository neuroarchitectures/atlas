# Architecture: ShuffleNet V1

## Motivation

Efficient CNN with pointwise group convolutions and channel shuffle for mobile.

## Core Idea

Efficient CNN with pointwise group convolutions and channel shuffle for mobile.

## Architecture

### Overview

![shufflenet-v1 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 224, 224] |
| 2 | Conv 1 (stride 2) | `conv2d` | outFeatures=24 |
| 3 | MaxPool (stride 2) | `maxpool2d` | outFeatures=24 |
| 4 | ShuffleNet Stage 2 | `conv2d` | outFeatures=232 |
| 5 | ShuffleNet Stage 3 | `conv2d` | outFeatures=464 |
| 6 | ShuffleNet Stage 4 | `conv2d` | outFeatures=1024 |
| 7 | Global Average Pool | `adaptiveavgpool2d` | outFeatures=1024 |
| 8 | FC Layer | `linear` | outFeatures=1000 |
| 9 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Conv 1 (stride 2)** — conv2d layer. 2. **MaxPool (stride 2)** — maxpool2d layer. 2. **ShuffleNet Stage 2** — conv2d layer. 2. **ShuffleNet Stage 3** — conv2d layer. 2. **ShuffleNet Stage 4** — conv2d layer. 2. **Global Average Pool** — adaptiveavgpool2d layer. 2. **FC Layer** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Efficient CNN with pointwise group convolutions and channel shuffle for mobile.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
