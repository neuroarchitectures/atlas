# Architecture: DARTS

## Motivation

NAS with RL or evolution is computationally expensive (thousands of GPU hours). DARTS makes the search differentiable by using a softmax over candidate operations, enabling gradient descent.

## Core Idea

Represent each edge as a weighted mixture of operations: o_bar(x) = sum_i alpha_i * o_i(x). Optimize alphas via gradient descent (bilevel optimization). At the end, discretize by keeping the best operation per edge.

## Architecture

### Overview

![darts architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [48, 48, 16] |
| 2 | Mixed Operation | `identity` | sum_i alpha_i * o_i(x) |
| 3 | Cell Output | `conv` | 1x1 conv to merge |
| 4 | Output | `output` |  |

</details>

### Components

1. **Continuous relaxation** — Each edge is a weighted sum of candidate operations (3x3 conv, 5x5 conv, max pool, avg pool, identity, zero). Weights are softmax-normalized. 2. **Bilevel optimization** — Alternates between updating network weights w (on training set) and architecture alphas (on validation set). 3. **Discretization** — After search, each edge is discretized to the operation with the highest alpha. 4. **Normal and reduction cells** — Two cell types are searched: normal (stride 1) and reduction (stride 2). 5. **Cell-based search space** — A DAG of nodes, where each node computes a sum of mixed operations from previous nodes.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Represent each edge as a weighted mixture of operations: o_bar(x) = sum_i alpha_i * o_i(x). Optimize alphas via gradient descent (bilevel optimization). At the end, discretize by keeping the best oper
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ENAS, NASNet. Successor: PC-DARTS, DARTS+, DrNAS.

## References

- Liu et al. 2019
