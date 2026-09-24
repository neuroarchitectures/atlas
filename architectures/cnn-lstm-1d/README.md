# 1D CNN + LSTM

## Overview

The standard hybrid for long 1D physiological signals (ECG, PPG, IMU): a Conv1D front-end extracts local morphology, an LSTM models long-range temporal context, a dense head classifies.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The standard hybrid for long 1D physiological signals (ECG, PPG, IMU): a Conv1D front-end extracts local morphology, an LSTM models long-range temporal context, a dense head classifies.

## Key Characteristics

- No single canonical paper: this is the pattern that appears in hundreds of biosignal papers, included as the reference baseline.
- The conv stem downsamples aggressively so the LSTM sees a short, feature-rich sequence instead of raw samples.

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
