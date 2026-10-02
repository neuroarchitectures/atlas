# Architecture: AudioLM

## Motivation

Generating coherent long-form audio requires both semantic continuity (what happens next) and acoustic fidelity (how it sounds). AudioLM models both hierarchically using a language model over discrete audio tokens.

## Core Idea

Audio is tokenized by SoundStream into discrete codes. A semantic language model (coarse tokens) generates the high-level structure. An acoustic language model (fine tokens) adds acoustic detail conditioned on semantic tokens. SoundStream decoder converts tokens back to waveform.

## Architecture

### Overview

![audiolm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Audio Waveform | `input` | 24kHz |
| 2 | SoundStream Tokenizer | `identity` | RVQ tokens |
| 3 | Semantic LM | `linear` | coarse tokens, Transformer |
| 4 | Acoustic LM | `linear` | fine tokens, Transformer |
| 5 | SoundStream Decoder | `identity` | tokens to waveform |
| 6 | Generated Audio | `output` | 24kHz |

</details>

### Components

1. **SoundStream tokenizer** — converts waveform to discrete tokens. 2. **Semantic LM** — autoregressive Transformer over coarse (first RVQ level) tokens. 3. **Acoustic LM** — autoregressive Transformer over fine (all RVQ levels) tokens, conditioned on semantic. 4. **Hierarchical generation** — semantic first, then acoustic. 5. **SoundStream decoder** — converts tokens back to waveform.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Audio is tokenized by SoundStream into discrete codes. A semantic language model (coarse tokens) generates the high-leve...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SoundStream (codec), Jukebox (hierarchical VQ-VAE). Successor: MusicLM, AudioGen.

## References

- Agarwal et al. 2022
