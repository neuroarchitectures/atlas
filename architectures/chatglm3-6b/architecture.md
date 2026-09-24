# Architecture: ChatGLM3-6B

## Motivation

The third generation of the ChatGLM series from Zhipu AI and Tsinghua, historically the most-starred Chinese open LLM. A pre-norm decoder with extreme grouped (near-multi-query) attention and GLM-style partial rotary embeddings.

## Core Idea

The third generation of the ChatGLM series from Zhipu AI and Tsinghua, historically the most-starred Chinese open LLM.

## Architecture

### Overview

![ChatGLM3-6B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 173 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.2B |
| Layers | 28 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 2 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 13,696 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
| Vocabulary | 65,024 |
| Max context | 8,192 |

`model.json` is the full 28-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face" (with importer fixes noted in the generator script), with all hyperparameters from the official `config.json`.

### Design Notes

- Aggressive multi-query-style attention: 32 query heads share only 2 KV groups (multi_query_group_num = 2), a 16:1 ratio that shrinks the KV cache far more than typical GQA.
- GLM-style partial RoPE: rotary embedding is applied to half of each 128-dim head (rotary dim 64); the other half stays position-free.
- The FFN intermediate size (13696) follows the GLM recipe rather than the Llama 8/3 ratio.
- 8192-token base context; -32k and -128k long-context variants exist. Input and output embeddings are untied (two 65024 x 4096 matrices).

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **6.24B**.
Hugging Face safetensors metadata reports **6.24B** for the real weights.
Deviation from the authoritative count (6.24B): **-0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Aggressive multi-query-style attention: 32 query heads share only 2 KV groups (multi_query_group_num = 2), a 16:1 ratio that shrinks the KV cache far more than typical GQA.
- GLM-style partial RoPE: rotary embedding is applied to half of each 128-dim head (rotary dim 64); the other half stays position-free.
- The FFN intermediate size (13696) follows the GLM recipe rather than the Llama 8/3 ratio.
- 8192-token base context; -32k and -128k long-context variants exist. Input and output embeddings are untied (two 65024 x 4096 matrices).

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

