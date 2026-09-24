# EEGNet

## Overview

The compact CNN that became the universal baseline for EEG brain-computer interfaces: temporal conv, depthwise spatial conv across electrodes, separable conv, all in a few thousand parameters.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The compact CNN that became the universal baseline for EEG brain-computer interfaces: temporal conv, depthwise spatial conv across electrodes, separable conv, all in a few thousand parameters.

## Key Characteristics

- Depthwise convolution across the electrode dimension learns spatial filters per temporal filter, mirroring classical CSP pipelines.
- Small enough to train on single-subject EEG datasets (hundreds of trials), which is the whole point.

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
