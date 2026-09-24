# Convolutional Positional Embedding

## Design Philosophy

Position information doesn't need parameters or tables: a small *depthwise convolution* over the token sequence leaks relative order into features by construction — local, translation-equivariant, and free of length limits.

## Functionality

- Latent sequence → grouped/depthwise 1D conv (kernel k, groups = channels) → result carries local relative-position information; replaces sinusoidal/learned PE (wav2vec 2.0).

## Used By

| Model | Role |
|-------|------|
| Wav2Vec2 base | Grouped 1D conv over the latent sequence before the Transformer |

## Features

- **Zero learned position table** — position is a convolution side-effect.
- **Local bias** — appropriate when relative locality matters more than absolute index (speech, time series).

## Evolution

- **Predecessor**: sinusoidal/learned absolute embeddings.
- **Related**: Mix-FFN (SegFormer) — position leakage via conv inside the FFN; CSWin's LePE (conv on V).
