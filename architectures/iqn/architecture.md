# Architecture: IQN

## Motivation

QR-DQN fixes quantile locations but cannot adjust them. A more flexible distributional approach is needed.

## Core Idea

Approximate the full quantile function Z(s,a) by sampling quantile fractions tau from [0,1] and learning the quantile values, enabling the model to represent the complete return distribution.

## Architecture

### Overview

![iqn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [4] |
| 2 | Conv2d | `conv2d` |  |
| 3 | Conv2d | `conv2d` |  |
| 4 | Conv2d | `conv2d` |  |
| 5 | Flatten | `flatten` |  |
| 6 | Feature | `linear` | outFeatures: 512 |
| 7 | Q-Values | `linear` | outFeatures: 4 |
| 8 | Action Values | `output` |  |

</details>

### Components

1. **Convolutional encoder** — CNN extracts features from game frames. 2. **Quantile embedding** — Cosine embedding of sampled tau values. 3. **Element-wise product** — Merge state features with quantile embeddings. 4. **Quantile output** — Predict quantile values for each action.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Approximate the full quantile function Z(s,a) by sampling quantile fractions tau from [0,1] and learning the quantile va
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: C51, QR-DQN. Successor: FQF, Fully Parameterized Quantile.

## References

Dabney et al. 2018
