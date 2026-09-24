# Conv1D Feature Extractor (Audio Stem)

## Design Philosophy

A stack of strided 1D convolutions that downsamples the raw waveform into latent frames before the Transformer. The philosophy: raw audio (16kHz) is too long for a Transformer (16000 samples/sec); the conv stem is the audio analogue of patch embedding — it turns the waveform into ~50 tokens/sec that the Transformer can process. Positions come from a depthwise conv (convolutional positional embedding), not sinusoids.

## Functionality

- **Input**: Raw 1D waveform `[1, T]` at 16kHz.
- **7 Conv1D layers**: Each with stride 2 (downsamples), kernel 10/3/3/3/3/2/2, width 512→512.
- **Output**: `[512, T/320]` — ~50 frames/sec, each 20ms of audio.
- **Normalization**: LayerNorm / group norm in the conv stack.
- **Positional embedding**: A depthwise Conv1D over the sequence (convolutional positional embedding).

## Used By

| Model | Role |
|-------|------|
| Wav2Vec2 | 7-layer conv feature extractor + 12-block Transformer |
| HuBERT | Same conv stem, self-supervised masked prediction |
| Whisper | 2-layer conv stem (80-mel input, 2× downsample) |

## Features

- **Waveform-to-tokens**: The bridge from raw audio to the Transformer's token world.
- **Learnable downsampling**: Strides are learned, not fixed.
- **Convolutional positions**: A depthwise conv provides relative position, not sinusoids.

## Evolution

- **Predecessor**: MFCC/spectrogram features (hand-engineered); raw-audio CNNs (sampleCNN).
- **Successor**: Audio transformers with conv stems are the standard; EnCodec uses a similar conv encoder for the codec.
