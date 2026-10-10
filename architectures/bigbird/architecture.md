# Architecture: BigBird

## Motivation

Long sequences need O(L) attention. BigBird combines three attention patterns: (1) random attention (each token attends to r random tokens), (2) window attention (local w neighbors), (3) global tokens (attend to/from all).

## Core Idea

The combination of random + window + global attention is a universal approximator of full attention (Turing complete) while being O(L) in complexity.

## Architecture

### Overview

![bigbird architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokens | `input` | shape: [4096, 512] |
| 2 | LayerNorm | `layerNorm` |  |
| 3 | QKV | `linear` | outFeatures: 1536 |
| 4 | Sparse Attention | `attention` | random + window + global |
| 5 | Residual | `add` |  |
| 6 | Output | `output` |  |

</details>

### Components

1. **Random attention** — Each token attends to r randomly selected tokens, providing long-range connectivity. 2. **Window attention** — Each token attends to its w nearest neighbors (local context). 3. **Global tokens** — A few designated tokens (e.g., CLS) attend to and are attended by all tokens. 4. **Turing completeness** — The combination is theoretically as expressive as full attention. 5. **Graph theory** — The random + window pattern ensures the attention graph is an expander, enabling information flow across the entire sequence.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — The combination of random + window + global attention is a universal approximator of full attention (Turing complete) while being O(L) in complexity.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Longformer, Sparse Transformer. Successor: BigBird-Pegasus, efficient long-context models.

## References

- Zaheer et al. 2020
