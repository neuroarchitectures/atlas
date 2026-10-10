# Architecture: Informer

## Motivation

Standard Transformers have O(L^2) attention complexity, limiting them to short sequences. Informer reduces this to O(L log L) while maintaining quality for long-horizon forecasting.

## Core Idea

Three key innovations: ProbSparse self-attention (keep only top-k active queries), self-attention distilling (downsample between layers), and generative-style decoder (one forward pass for entire horizon).

## Architecture

### Overview

![informer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input Sequence | `input` | shape: [96, 8] (L=96 lookback) |
| 2 | Embedding | `embedding` | dimension: 512 |
| 3 | ProbSparse Attention | `attention` | O(L log L) complexity |
| 4 | Distilling (MaxPool) | `identity` | stride 2, downsample |
| 5 | ProbSparse Attention 2 | `attention` | reduced length |
| 6 | Generative Decoder | `linear` | one-pass horizon prediction |
| 7 | Forecast | `output` | shape: [24] |

</details>

### Components

1. **ProbSparse self-attention** — Computes attention only for the top-k queries with highest KL divergence from uniform attention. Queries with low attention entropy (active queries) are kept; lazy queries use uniform attention. 2. **Self-attention distilling** — MaxPool with stride 2 between attention layers, halving the sequence length. 3. **Generative-style decoder** — Uses a start token and predicts the entire horizon in one forward pass, avoiding autoregressive error accumulation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Three key innovations: ProbSparse self-attention (keep only top-k active queries), self-attention distilling (downsample between layers), and generative-style decoder (one forward pass for entire hori
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Transformer (full attention). Successor: Autoformer, FEDformer, PatchTST.

## References

- Zhou et al. 2021
