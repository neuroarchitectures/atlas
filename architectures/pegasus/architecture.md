# Architecture: PEGASUS

## Motivation

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization: gap-sentence masking for summarization.

## Core Idea

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization: gap-sentence masking for summarization.

## Architecture

### Overview

![pegasus architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[1024] |
| 2 | Token Embedding | `embedding` | vocabSize=96103, dim=1024 |
| 3 | Transformer Encoder (16 layers) | `transformer-encoder` | numLayers=16, hiddenSize=1024, numHeads=16 |
| 4 | Transformer Decoder (16 layers) | `transformer-decoder` | numLayers=16, hiddenSize=1024, numHeads=16 |
| 5 | LM Head | `linear` | outFeatures=96103 |
| 6 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Encoder (16 layers)** — transformer-encoder layer. 2. **Transformer Decoder (16 layers)** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Pre-training with Extracted Gap-sentences for Abstractive SUmmarization: gap-sentence masking for summarization.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
