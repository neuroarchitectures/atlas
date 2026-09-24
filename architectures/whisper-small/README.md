# Whisper Small

## Overview

OpenAI's speech recognition encoder-decoder: log-mel spectrogram through a two-layer conv stem into a 12-block Transformer encoder, with a 12-block causal text decoder cross-attending to the audio states.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

OpenAI's speech recognition encoder-decoder: log-mel spectrogram through a two-layer conv stem into a 12-block Transformer encoder, with a 12-block causal text decoder cross-attending to the audio states.

## Key Characteristics

- The conv stem (two 1D convs with GeLU) downsamples the 80-channel mel input 2x before the encoder; everything after is a vanilla Transformer.
- Trained on 680k hours of weakly supervised audio; the architecture is intentionally boring so the data does the work.
- The full graph carries all 12 encoder blocks and 12 decoder blocks, each decoder block with its own cross-attention into the encoder output.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Speech-to-text encoder-decoder |
| Parameters | 244M |
| Layers | 12 encoder + 12 decoder |
| Hidden size | 768 |
| Attention | 12 heads; decoder adds cross-attention |
| FFN | Dense MLP, 3072, GeLU |
| Audio stem | Two Conv1D layers (80-mel input, 2x downsample) |
| Positions | Sinusoidal (encoder), learned (decoder) |
| Vocabulary | 51,865 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** MIT. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
