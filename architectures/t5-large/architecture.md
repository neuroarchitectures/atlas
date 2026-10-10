# Architecture: T5-Large

## Motivation

NLP tasks use different model architectures (classification, generation, QA). A unified framework would simplify deployment.

## Core Idea

Cast all NLP tasks as text-to-text: input task prefix + text, output target text. Scale up the encoder-decoder transformer.

## Architecture

### Overview

![t5-large architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape: [512] |
| 2 | Token Embedding | `embedding` | vocab: 32128 |
| 3 | Encoder | `transformer` | 24 layers |
| 4 | Decoder | `transformer` | 24 layers |
| 5 | Output Head | `linear` | vocab: 32128 |
| 6 | Output Tokens | `output` |  |

</details>

### Components

1. **Token embedding** — 32K vocabulary, 1024-dim. 2. **Encoder** — 24-layer bidirectional transformer. 3. **Decoder** — 24-layer autoregressive transformer with cross-attention. 4. **Text-to-text** — Task prefix (e.g., 'translate English to German:') conditions the model. 5. **Output head** — Linear projection to vocabulary.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Cast all NLP tasks as text-to-text: input task prefix + text, output target text. Scale up the encoder-decoder transform
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: T5-Small, BERT. Successor: T5-11B, FLAN-T5.

## References

Raffel et al. 2020
