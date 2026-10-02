# Architecture: BLOOM

## Motivation

BigScience Large Open-science Open-access Multilingual Language Model: 176B decoder-only transformer.

## Core Idea

BigScience Large Open-science Open-access Multilingual Language Model: 176B decoder-only transformer.

## Architecture

### Overview

![bloom architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[2048] |
| 2 | Token Embedding | `embedding` | vocabSize=250880, dim=14336 |
| 3 | Transformer Decoder (70 layers) | `transformer-decoder` | numLayers=70, hiddenSize=14336, numHeads=112 |
| 4 | LM Head | `linear` | outFeatures=250880 |
| 5 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Decoder (70 layers)** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — BigScience Large Open-science Open-access Multilingual Language Model: 176B decoder-only transformer.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
