# Architecture: RoBERTa

## Motivation

BERT was a breakthrough but its pre-training recipe was suboptimal: static masking, small batches, and limited training data.

## Core Idea

Optimize BERT's pre-training with dynamic masking, larger batch sizes, longer training, more data, and removal of next sentence prediction.

## Architecture

### Overview

![roberta architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape: [512] |
| 2 | Token Embedding | `embedding` | vocab: 50265 |
| 3 | Positional Encoding | `positional-encoding` | learned |
| 4 | Transformer Encoder | `transformer` | 24 layers |
| 5 | LayerNorm | `layernorm` |  |
| 6 | LM Head | `linear` | vocab: 50265 |
| 7 | Token Probabilities | `output` |  |

</details>

### Components

1. **Token embedding** — 50K vocabulary, 768-dim. 2. **Dynamic masking** — Different masks each epoch. 3. **Transformer encoder** — 24 layers (Large). 4. **No NSP** — Next sentence prediction removed. 5. **MLM head** — Masked language modeling with layer norm.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Optimize BERT's pre-training with dynamic masking, larger batch sizes, longer training, more data, and removal of next s
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: BERT. Successor: DeBERTa, ELECTRA.

## References

Liu et al. 2019
