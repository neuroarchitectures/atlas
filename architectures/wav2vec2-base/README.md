# Wav2Vec2 base

## Overview

The model that made self-supervised speech work: a convolutional feature extractor turns the raw 16kHz waveform into latent frames, a Transformer contextualizes them, and contrastive masked prediction over a learned codebook does the pretraining. Fine-tune with a tiny head and it beats systems trained on 100x the labels.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The model that made self-supervised speech work: a convolutional feature extractor turns the raw 16kHz waveform into latent frames, a Transformer contextualizes them, and contrastive masked prediction over a learned codebook does the pretraining.

## Key Characteristics

- The 7-layer conv stack is the audio analogue of patch embedding: it downsamples raw audio to ~50 frames/sec before the Transformer ever sees it.
- Positions come from a depthwise conv over the sequence (a convolutional positional embedding), not sinusoids.
- Compare with [whisper-small](../whisper-small/): both put a conv stem before a Transformer, but Whisper is supervised seq2seq while Wav2Vec2 is self-supervised representation learning.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Self-supervised speech encoder |
| Parameters | 95M |
| Feature extractor | 7 Conv1D layers (raw waveform → 20ms frames) |
| Encoder | 12 Transformer blocks, 768 hidden, 12 heads |
| Positions | Convolutional positional embedding |
| Pretraining | Contrastive masked prediction over quantized latents |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
