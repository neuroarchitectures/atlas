# Architecture: LoRA

## Motivation

Fine-tuning large language models updates all parameters, which is expensive and leads to catastrophic forgetting. LoRA freezes the model and adds small trainable low-rank matrices.

## Core Idea

For each weight matrix W, add delta W = A * B where A is d x r and B is r x k (r << min(d,k)). Only A and B are trained. At inference, W_eff = W + A * B with no additional latency.

## Architecture

### Overview

![lora architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1024] |
| 2 | Frozen Weight W | `linear` | not updated, 1024x1024 |
| 2 | LoRA A (d x r) | `linear` | outFeatures: 8 (rank r=8) |
| 3 | LoRA B (r x k) | `linear` | outFeatures: 1024, inFeatures: 8 |
| 4 | Merge | `add` | W + A*B |
| 5 | Output | `output` | h = Wx + BAx |

</details>

### Components

1. **Low-rank decomposition** — Delta W = A * B, where A is initialized with Gaussian and B with zeros (so delta W starts as 0). 2. **Rank r** — Typically 4, 8, or 16. Much smaller than the original dimension. 3. **No inference overhead** — At inference, W_eff = W + A * B can be pre-merged, so there's no additional computation. 4. **Multiple LoRA** — Different LoRA modules can be trained for different tasks and swapped at inference. 5. **Applied to attention** — Typically applied to Q and V projection matrices in attention layers.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — For each weight matrix W, add delta W = A * B where A is d x r and B is r x k (r << min(d,k)). Only A and B are trained. At inference, W_eff = W + A * B with no additional latency.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Adapter, Prefix Tuning. Successor: QLoRA, DoRA, LoRA variants.

## References

- Hu et al. 2021
