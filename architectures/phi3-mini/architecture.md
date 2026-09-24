# Architecture: Phi-3 Mini Block

## Motivation

The Phi-3 Mini (3.8B) decoder block: a compact Llama-style block at 3072 hidden, the architecture behind the "small model trained on textbook-quality data" line of work.

## Core Idea

The Phi-3 Mini (3.

## Architecture

### Overview

![Phi-3 Mini Block architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | tokens | `input` | shape: [1, 2048] |
| 2 | embed | `embedding` | numEmbeddings: 32064, embeddingDim: 3072 |
| 3 | rope | `rope` | headDim: 96 |
| 4 | norm_attn | `rmsNorm` | normalizedShape: 3072 |
| 5 | attn | `groupedQueryAttention` | embedDim: 3072, numHeads: 32, numKVHeads: 32 |
| 6 | residual_1 | `add` |   |
| 7 | norm_ffn | `rmsNorm` | normalizedShape: 3072 |
| 8 | ffn | `swiglu` | embedDim: 3072, intermediateSize: 8192 |
| 9 | residual_2 | `add` |   |
| 10 | norm_out | `rmsNorm` | normalizedShape: 3072 |
| 11 | lm_head | `linear` | outFeatures: 32064, inFeatures: 3072 |
| 12 | output | `output` |   |

</details>

This graph ships in Neurarch's in-app template library; the copy here passes shape propagation with zero errors.

### Design Notes

- Full multi-head attention (32 heads at 3072, head dim 96); no GQA at this scale in the 4k variant.
- SwiGLU FFN at 8192 intermediate, RMSNorm pre-norm, RoPE.
- Architecturally conventional on purpose: the Phi thesis is that data quality, not architecture novelty, drives small-model performance.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Full multi-head attention (32 heads at 3072, head dim 96); no GQA at this scale in the 4k variant.
- SwiGLU FFN at 8192 intermediate, RMSNorm pre-norm, RoPE.
- Architecturally conventional on purpose: the Phi thesis is that data quality, not architecture novelty, drives small-model performance.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` holds all nodes at real dimensions.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

