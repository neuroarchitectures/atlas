# Architecture: XLNet

## Motivation

BERT's MLM assumes independence between masked tokens and cannot model dependencies. Autoregressive models are unidirectional.

## Core Idea

Use permutation language modeling: predict tokens in random order, enabling bidirectional context while maintaining autoregressive factorization.

## Architecture

### Overview

![xlnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape: [512] |
| 2 | Token Embedding | `embedding` | vocab: 32000 |
| 3 | Relative Positional | `positional-encoding` | relative |
| 4 | Transformer-XL | `transformer-xl` | 24 layers |
| 5 | Output Head | `linear` | vocab: 32000 |
| 6 | Token Probabilities | `output` |  |

</details>

### Components

1. **Token embedding** — Standard token embeddings. 2. **Relative positional encoding** — From Transformer-XL, enables long context. 3. **Two-stream attention** — Content stream (full context) and query stream (position only). 4. **Permutation LM** — Factorize over random permutations of the token sequence. 5. **Segment recurrence** — Memory from previous segments.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use permutation language modeling: predict tokens in random order, enabling bidirectional context while maintaining auto
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: BERT, Transformer-XL. Successor: RoBERTa, ELECTRA.

## References

Yang et al. 2019
