# Architecture: Inception-v4

## Motivation

Refined Inception architecture with uniform stem, Inception-A/B/C blocks and reduction modules.

## Core Idea

Refined Inception architecture with uniform stem, Inception-A/B/C blocks and reduction modules.

## Architecture

### Overview

![inception-v4 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 299, 299] |
| 2 | Stem Block | `conv2d` | outFeatures=384 |
| 3 | Inception-A x4 | `conv2d` | outFeatures=384 |
| 4 | Reduction-A | `conv2d` | outFeatures=1024 |
| 5 | Inception-B x7 | `conv2d` | outFeatures=1024 |
| 6 | Reduction-B | `conv2d` | outFeatures=1536 |
| 7 | Inception-C x3 | `conv2d` | outFeatures=1536 |
| 8 | Global Average Pool | `adaptiveavgpool2d` | outFeatures=1536 |
| 9 | FC Layer | `linear` | outFeatures=1000 |
| 10 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **Stem Block** — conv2d layer. 2. **Inception-A x4** — conv2d layer. 2. **Reduction-A** — conv2d layer. 2. **Inception-B x7** — conv2d layer. 2. **Reduction-B** — conv2d layer. 2. **Inception-C x3** — conv2d layer. 2. **Global Average Pool** — adaptiveavgpool2d layer. 2. **FC Layer** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Refined Inception architecture with uniform stem, Inception-A/B/C blocks and reduction modules.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
