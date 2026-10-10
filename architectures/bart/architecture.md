# Architecture: BART

## Motivation

Bidirectional encoder + autoregressive decoder seq2seq model for denoising autoencoding pre-training.

## Core Idea

Bidirectional encoder + autoregressive decoder seq2seq model for denoising autoencoding pre-training.

## Architecture

### Overview

![bart architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[1024] |
| 2 | BART Encoder | `transformer-encoder` | numLayers=6, hiddenSize=1024, numHeads=16 |
| 3 | BART Decoder | `transformer-decoder` | numLayers=6, hiddenSize=1024, numHeads=16 |
| 4 | LM Head | `linear` | outFeatures=50265 |
| 5 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **BART Encoder** — transformer-encoder layer. 2. **BART Decoder** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Bidirectional encoder + autoregressive decoder seq2seq model for denoising autoencoding pre-training.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
