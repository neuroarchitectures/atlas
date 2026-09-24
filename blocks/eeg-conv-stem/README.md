# EEG Conv Tokenizer Stem

## Design Philosophy

Raw EEG is low-SNR, electrode-structured, and non-stationary. A specialized stem converts it to features: temporal convolution extracts waveform dynamics, *depthwise convolution across electrodes* learns CSP-like spatial filters, and pooling stabilizes — producing tokens for downstream attention.

## Functionality

- EEGNet: 1×64 temporal conv (8 filters) → 22×1 depthwise spatial conv ×2 (16 filters) → separable 1×16 conv → pooling.
- EEG-Conformer reuses the stem, then feeds 40-dim tokens to 10-head attention.

## Used By

| Model | Role |
|-------|------|
| EEGNet | Compact end-to-end EEG classification |
| EEG Conformer | Tokenizer stem feeding the transformer encoder |

## Features

- **Spatial filters are learned** — data-driven electrode weighting (CSP analog).
- **Few parameters** — stem dominates the compact model's efficiency.

## Evolution

- **Predecessor**: hand-crafted CSP + bandpower feature pipelines.
- **Related**: conv1d-audio-stem — the audio-signal sibling; patch-embedding for images.
