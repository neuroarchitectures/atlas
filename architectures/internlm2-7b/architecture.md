# Architecture: InternLM2-7B

## Motivation

Shanghai AI Laboratory's second-generation 7B base model. A pre-norm decoder with 32:8 grouped-query attention and a native 32k context window, plus strong tool-use and long-context chat variants.

## Core Idea

Shanghai AI Laboratory's second-generation 7B base model.

## Architecture

### Overview

![InternLM2-7B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 197 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.7B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 92,544 |
| Max context | 32,768 |

`model.json` is the full 32-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- Llama-3-like shape a year early: 32 layers, GQA 32:8, and the same 14336 FFN intermediate size.
- Native 32768-token context trained with rope_theta = 1e6, extrapolating to 200k in the chat variants.
- Consolidated W_qkv weight layout: q, k, v are stored interleaved per KV group, which matters when converting checkpoints.
- 92544-token vocabulary covering Chinese, English, and code.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **7.74B**.
Deviation from the authoritative count (7.74B): **-0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Llama-3-like shape a year early: 32 layers, GQA 32:8, and the same 14336 FFN intermediate size.
- Native 32768-token context trained with rope_theta = 1e6, extrapolating to 200k in the chat variants.
- Consolidated W_qkv weight layout: q, k, v are stored interleaved per KV group, which matters when converting checkpoints.
- 92544-token vocabulary covering Chinese, English, and code.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

