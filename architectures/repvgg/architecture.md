# Architecture: RepVGG

## Motivation

VGG-style CNN with re-parameterization: multi-branch training collapsed to single conv for fast inference.

## Core Idea

VGG-style CNN with re-parameterization: multi-branch training collapsed to single conv for fast inference.

## Architecture

### Overview

![repvgg architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 224, 224] |
| 2 | Stage 1 (stride 2) | `conv2d` | outFeatures=64 |
| 3 | Stage 2 (stride 2) | `conv2d` | outFeatures=128 |
| 4 | Stage 3 (stride 2) | `conv2d` | outFeatures=256 |
| 5 | Stage 4 (stride 2) | `conv2d` | outFeatures=512 |
| 6 | Stage 5 (stride 2) | `conv2d` | outFeatures=512 |
| 7 | Global Average Pool | `adaptiveavgpool2d` | outFeatures=512 |
| 8 | FC Layer | `linear` | outFeatures=1000 |
| 9 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Stage 1 (stride 2)** — conv2d layer. 2. **Stage 2 (stride 2)** — conv2d layer. 2. **Stage 3 (stride 2)** — conv2d layer. 2. **Stage 4 (stride 2)** — conv2d layer. 2. **Stage 5 (stride 2)** — conv2d layer. 2. **Global Average Pool** — adaptiveavgpool2d layer. 2. **FC Layer** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — VGG-style CNN with re-parameterization: multi-branch training collapsed to single conv for fast inference.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
