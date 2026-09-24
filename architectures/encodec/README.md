# EnCodec

## Overview

A neural audio codec: a convolutional encoder downsamples the waveform, residual vector quantization turns the latents into a stack of discrete codes, and a mirrored decoder reconstructs the audio. The thing that turns continuous audio into tokens an LLM can model.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A neural audio codec: a convolutional encoder downsamples the waveform, residual vector quantization turns the latents into a stack of discrete codes, and a mirrored decoder reconstructs the audio.

## Key Characteristics

- Residual vector quantization (RVQ): a cascade of codebooks where each quantizes the residual the previous one left, so a few codebooks reconstruct audio at very low bitrate.
- The encoder/decoder are SEANet-style strided conv stacks with an LSTM in the bottleneck; the discrete codes are what MusicGen / audio LLMs actually generate.
- Conceptually the audio analogue of a VAE+codebook (VQ-VAE) for images; it is why "audio language model" is even possible.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Neural audio codec (autoencoder + quantizer) |
| Encoder | Strided Conv1D stack (downsamples the waveform) |
| Bottleneck | Residual vector quantization (discrete codes) |
| Decoder | Mirrored transposed-conv stack (reconstructs audio) |
| Use | The audio tokenizer behind audio LLMs (MusicGen, ...) |

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
