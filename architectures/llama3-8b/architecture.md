# Architecture: Llama-3-8B

## Motivation

Meta's 8B dense decoder, the de facto baseline architecture that most current open LLMs (including many in this zoo) derive from or compare against.

## Core Idea

Meta's 8B dense decoder, the de facto baseline architecture that most current open LLMs (including many in this zoo) derive from or compare against.

## Architecture

### Overview

![Llama-3-8B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 197 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 8B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 128,256 |
| Max context | 8,192 |

`model.json` is the full 32-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- The reference open-weight architecture of 2024: 32 layers, GQA 32:8, SwiGLU 14336, RMSNorm pre-norm. Half the models in this zoo are best described as deltas against this graph.
- Big vocabulary jump over Llama-2: 128256 tokens (tiktoken-style BPE), which moves a meaningful fraction of parameters into the embedding and head.
- rope_theta raised to 500000 for the native 8192-token context.
- Graph imported via the NousResearch mirror because the official repo is gated; the config is byte-identical.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **8.03B**.
Hugging Face safetensors metadata reports **8.03B** for the real weights.
Deviation from the authoritative count (8.03B): **+0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- The reference open-weight architecture of 2024: 32 layers, GQA 32:8, SwiGLU 14336, RMSNorm pre-norm. Half the models in this zoo are best described as deltas against this graph.
- Big vocabulary jump over Llama-2: 128256 tokens (tiktoken-style BPE), which moves a meaningful fraction of parameters into the embedding and head.
- rope_theta raised to 500000 for the native 8192-token context.
- Graph imported via the NousResearch mirror because the official repo is gated; the config is byte-identical.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

