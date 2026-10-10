# Architecture: OFA

## Motivation

Unified seq2seq pre-training for multimodal tasks: single encoder-decoder for image, text, and grounded tasks.

## Core Idea

Unified seq2seq pre-training for multimodal tasks: single encoder-decoder for image, text, and grounded tasks.

## Architecture

### Overview

![ofa architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokenized Input | `input` | shape=[1024] |
| 2 | Token Embedding | `embedding` | vocabSize=64000, dim=1024 |
| 3 | Transformer Encoder | `transformer-encoder` | numLayers=12, hiddenSize=1024, numHeads=16 |
| 4 | Transformer Decoder | `transformer-decoder` | numLayers=12, hiddenSize=1024, numHeads=16 |
| 5 | LM Head | `linear` | outFeatures=64000 |
| 6 | Output Sequence | `output` | — |

</details>

### Components

1. **Tokenized Input** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Encoder** — transformer-encoder layer. 2. **Transformer Decoder** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Sequence** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Unified seq2seq pre-training for multimodal tasks: single encoder-decoder for image, text, and grounded tasks.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
