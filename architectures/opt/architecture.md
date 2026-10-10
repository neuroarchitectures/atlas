# Architecture: OPT

## Motivation

Open Pre-trained Transformer: open-source decoder-only LLM matching GPT-3 scale (125M–175B).

## Core Idea

Open Pre-trained Transformer: open-source decoder-only LLM matching GPT-3 scale (125M–175B).

## Architecture

### Overview

![opt architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[2048] |
| 2 | Token Embedding | `embedding` | vocabSize=50272, dim=2048 |
| 3 | Transformer Decoder (24 layers) | `transformer-decoder` | numLayers=24, hiddenSize=2048, numHeads=16 |
| 4 | LM Head | `linear` | outFeatures=50272 |
| 5 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Decoder (24 layers)** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Open Pre-trained Transformer: open-source decoder-only LLM matching GPT-3 scale (125M–175B).
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
