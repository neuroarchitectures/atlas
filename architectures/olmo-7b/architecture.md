# Architecture: OLMo-7B

## Motivation

Ai2's fully-open 7B: not just open weights but open training data (Dolma), code, and logs. A Llama-shaped decoder with one signature twist, non-parametric LayerNorm, included as the reference for reproducible LLM research.

## Core Idea

Ai2's fully-open 7B: not just open weights but open training data (Dolma), code, and logs.

## Architecture

### Overview

![OLMo-7B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 197 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.9B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Multi-head: 32 heads |
| FFN | SwiGLU, intermediate size 11008 |
| Normalization | Non-parametric LayerNorm (no weight / bias) |
| Positions | RoPE |
| Vocabulary | 50,304 |
| Max context | 4,096 |

`model.json` is the full graph (repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face".

### Design Notes

- Non-parametric LayerNorm: the norm has no learnable gain or bias at all, just the normalization, which the OLMo report found improved stability.
- Otherwise a clean Llama-style decoder: RoPE, SwiGLU, no biases, untied embeddings.
- The point is end-to-end openness (Dolma corpus + training code + checkpoints), making it the model to use when you need to know exactly what went in.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **6.89B**.
Hugging Face safetensors metadata reports **6.89B** for the real weights.
Deviation from the authoritative count (6.89B): **+0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Non-parametric LayerNorm: the norm has no learnable gain or bias at all, just the normalization, which the OLMo report found improved stability.
- Otherwise a clean Llama-style decoder: RoPE, SwiGLU, no biases, untied embeddings.
- The point is end-to-end openness (Dolma corpus + training code + checkpoints), making it the model to use when you need to know exactly what went in.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

