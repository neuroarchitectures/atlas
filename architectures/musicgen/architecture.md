# Architecture: MusicGen

## Motivation

Generating music requires modeling long-range temporal dependencies and multi-instrumental structure. MusicGen uses a compressed discrete audio representation (EnCodec) and a transformer for efficient music generation.

## Core Idea

Encode audio with EnCodec (4 codebooks at 50 Hz). Train a transformer decoder to predict the next token. Use delayed inter-codebook prediction pattern and chromagram conditioning for melody control.

## Architecture

### Overview

![musicgen architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text Prompt | `input` | shape: [256] |
| 2 | Text Encoder | `embedding` | T5-based text encoder |
| 3 | Transformer Decoder | `linear` | 32 layers, 1024 dim, 8 heads |
| 4 | LM Head | `linear` | predict next EnCodec token |
| 5 | EnCodec Decode | `identity` | decode tokens to waveform |
| 6 | Audio | `output` |  |

</details>

### Components

1. **EnCodec tokenizer** — Compresses audio to 4 codebooks at 50 Hz (1500 tokens/min). 2. **Transformer decoder** — 32-layer decoder-only transformer (1.3B/3.3B/6.7B params). Predicts EnCodec tokens autoregressively. 3. **Delayed pattern** — Codebooks are predicted with a delay, reducing the effective sequence length. 4. **Chroma conditioning** — Optional chromagram input for melody control. 5. **Classifier-free guidance** — Text prompt dropout during training for controllable generation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Encode audio with EnCodec (4 codebooks at 50 Hz). Train a transformer decoder to predict the next token. Use delayed inter-codebook prediction pattern and chromagram conditioning for melody control.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AudioLM, EnCodec. Successor: MusicGen variants, AudioCraft.

## References

- Copet et al. 2024
