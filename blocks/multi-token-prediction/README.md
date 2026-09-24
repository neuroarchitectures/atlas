# Multi-Token Prediction (MTP)

## Design Philosophy

One-token-ahead supervision wastes most of the forward pass's information. Predict *k future tokens* with extra lightweight MTP modules during training — denser supervision signals improve planning-like representations — and discard the modules at inference, where they optionally enable speculative decoding.

## Functionality

- Per MTP depth i: shared trunk hidden state + small head predicting token t+i+1; losses summed over depths.
- DeepSeek-V3: 2 MTP modules on the 7168-dim hidden; optional self-speculative decoding reuses them.

## Used By

| Model | Role |
|-------|------|
| DeepSeek-V3 | Training-only MTP heads; speculative decoding at inference |

## Features

- **Denser training signal** — k extra supervisions per position at low cost.
- **Free speculative decoding** — the trained heads double as drafters.

## Evolution

- **Predecessor**: Gloeckle et al. 2024 multi-token prediction; Meten.
- **Related**: continuous-embedding-prediction — predicting future *embeddings* instead of tokens; block-causal-diffusion (parallelism at generation instead).
