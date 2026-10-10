# Architecture: ELECTRA

## Motivation

MLM (BERT) only trains on masked tokens (~15%), wasting compute on unmasked tokens. A more sample-efficient objective is needed.

## Core Idea

Use replaced token detection: a small generator corrupts tokens, and a discriminator predicts whether each token is original or replaced. All tokens contribute to the loss.

## Architecture

### Overview

![electra architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape: [128] |
| 2 | Token Embedding | `embedding` | vocab: 30522 |
| 3 | Generator | `transformer` | 12 layers |
| 4 | Discriminator | `transformer` | 12 layers |
| 5 | Binary Head | `linear` | outFeatures: 2 |
| 6 | Replaced / Original | `output` |  |

</details>

### Components

1. **Generator** — Small MLM that proposes token replacements. 2. **Discriminator** — Binary classifier on each token (original vs. replaced). 3. **Shared embeddings** — Generator and discriminator share token embeddings. 4. **Training** — Only discriminator used for downstream tasks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use replaced token detection: a small generator corrupts tokens, and a discriminator predicts whether each token is orig
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: BERT, RoBERTa. Successor: DeBERTa.

## References

Clark et al. 2020
