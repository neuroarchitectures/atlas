# Architecture: Bark

## Motivation

Generating diverse audio (speech, music, sound effects) from text requires a model that can handle multiple audio modalities.

## Core Idea

Use a hierarchical transformer that generates audio codec tokens in coarse-to-fine stages, enabling text-to-audio generation without explicit vocoding.

## Architecture

### Overview

![bark architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text | `input` |  |
| 2 | Text Transformer | `transformer` | 12 layers |
| 3 | Coarse Audio Transformer | `transformer` | 12 layers |
| 4 | Fine Audio Transformer | `transformer` | 12 layers |
| 5 | Audio Tokens | `output` |  |

</details>

### Components

1. **Text transformer** — Encodes text prompt into semantic tokens. 2. **Coarse audio transformer** — Generates coarse EnCodec tokens conditioned on text. 3. **Fine audio transformer** — Refines coarse tokens to full-fidelity codec tokens. 4. **EnCodec decoder** — Converts tokens to waveform.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a hierarchical transformer that generates audio codec tokens in coarse-to-fine stages, enabling text-to-audio genera
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AudioLM, MusicGen. Successor: AudioLDM, Stable Audio.

## References

Suno 2023
