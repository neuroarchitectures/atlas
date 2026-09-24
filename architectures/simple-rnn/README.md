# Simple RNN

## Overview

A minimal Elman-style recurrent network for sequence processing. Four nodes: the smallest sequential model in the zoo, kept as a teaching baseline.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A minimal Elman-style recurrent network for sequence processing.

## Key Characteristics

- The architecture every LSTM, GRU, and ultimately attention mechanism was reacting to.
- Useful as a first graph for learning the canvas: add layers, watch shapes propagate.

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
