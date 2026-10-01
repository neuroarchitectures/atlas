# Architecture: Capsule Network

## Motivation

CNNs lose spatial hierarchies through pooling. Capsule Networks represent entities as vectors (capsules) whose direction encodes instantiation parameters and whose magnitude encodes existence probability.

## Core Idea

Replace scalar neurons with capsule vectors. Use dynamic routing-by-agreement: lower-level capsules send their outputs to higher-level capsules that agree with them, determined iteratively.

## Architecture

### Overview

![capsule network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [1, 28, 28] |
| 2 | Conv1 | `conv` | outChannels: 256, kernelSize: 9 |
| 3 | ReLU | `relu` |  |
| 4 | Primary Capsules | `linear` | 32 capsules x 8-dim, 256 channels |
| 5 | Squash | `identity` | ||s||^2/(1+||s||^2) * s/||s|| |
| 6 | Dynamic Routing | `attention` | 3 routing iterations |
| 7 | Digit Capsules | `output` | 10 capsules x 16-dim |

</details>

### Components

1. **Capsule** — A vector whose magnitude represents the probability of an entity's existence and whose direction represents instantiation parameters (pose, size, deformation). 2. **Squash function** — Non-linear squashing: v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||, ensuring short vectors get shrunk to near zero. 3. **Dynamic routing-by-agreement** — Lower capsules send outputs to higher capsules proportional to agreement (dot product). Routing is iterated 3 times. 4. **Reconstruction** — A decoder reconstructs the image from the digit capsule, serving as a regularizer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace scalar neurons with capsule vectors. Use dynamic routing-by-agreement: lower-level capsules send their outputs to higher-level capsules that agree with them, determined iteratively.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CNN. Successor: Capsule Network variants, EM routing capsules.

## References

- Sabour et al. 2017
