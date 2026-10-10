# Architecture: Tacotron 2

## Motivation

Neural end-to-end TTS needed to bridge text and audio without hand-engineered linguistic features. Tacotron 2 showed that an attention-based seq2seq encoder-decoder generating mel-spectrograms, paired with a neural vocoder, produces natural-sounding speech.

## Core Idea

Character embeddings pass through a convolutional bank + highway network and a bidirectional LSTM encoder. An attention-based LSTM decoder generates mel-spectrogram frames autoregressively. A WaveNet vocoder converts mel-spectrograms to waveforms.

## Architecture

### Overview

![tacotron2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text Characters | `input` | 256 chars |
| 2 | Character Embedding | `linear` | 256 dim |
| 3 | Conv Bank + Highway | `conv1d` | banks + projection |
| 4 | Bi-LSTM Encoder | `lstm` | hidden=256, bidirectional |
| 5 | Attention Decoder | `lstm` | 2 layers, attention |
| 6 | Mel Spectrogram | `linear` | 80 mel bins |
| 7 | Mel Output | `output` | mel frames |

</details>

### Components

1. **Character embedding** — learned embeddings for each character. 2. **Convolutional bank** — multiple filter widths capture n-gram-like patterns. 3. **Highway network** — enables information flow across layers. 4. **Bi-LSTM encoder** — captures context from both directions. 5. **Location-sensitive attention** — cumulative attention weights prevent skipping/repeating. 6. **Autoregressive decoder** — generates one mel frame per step. 7. **Post-net** — CNN refines mel-spectrogram. 8. **WaveNet vocoder** — neural vocoder converts mel to waveform.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Character embeddings pass through a convolutional bank + highway network and a bidirectional LSTM encoder. An attention-...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Tacotron (2017), Deep Voice. Successor: FastSpeech (non-autoregressive), VITS (end-to-end).

## References

- Shen et al. 2018
