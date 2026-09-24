# EEG Conformer

## Overview

Convolution meets attention for EEG decoding: an EEGNet-style conv stem tokenizes the signal, then a Transformer encoder models long-range temporal dependencies. State of the art on motor-imagery benchmarks.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Convolution meets attention for EEG decoding: an EEGNet-style conv stem tokenizes the signal, then a Transformer encoder models long-range temporal dependencies.

## Key Characteristics

- The conv stem does what patching does for ViT: turns raw 1000Hz multichannel EEG into a short token sequence the Transformer can afford.
- Pairs with [eegnet](../eegnet/) as the "before and after attention" story for biosignal models.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

## Related Architectures

See `references/README.md` for related architectures and research context.
