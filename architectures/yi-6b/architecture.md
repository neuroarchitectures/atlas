# Architecture: Yi-6B

## Motivation

The 6B base model from 01.AI (founded by Kai-Fu Lee), trained on 3.1T bilingual tokens. A Llama-style decoder notable for using grouped-query attention at small scale and for its 200K long-context variant.

## Core Idea

The 6B base model from 01.

## Architecture

### Overview

![Yi-6B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 197 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 4 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 11,008 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 64,000 |
| Max context | 4,096 |

`model.json` is the full 32-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- Fully Llama-compatible architecture (model_type "llama"), so the whole Llama tooling ecosystem works out of the box.
- Grouped-query attention even at 6B scale: 32 query heads over 4 KV heads, unusual for a model this small at the time.
- High rope_theta (5e6) chosen with long-context extension in mind; the Yi-6B-200K variant stretches the same architecture to 200k tokens.
- Compact 64000-token bilingual vocabulary.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **6.06B**.
Hugging Face safetensors metadata reports **6.06B** for the real weights.
Deviation from the authoritative count (6.06B): **+0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Fully Llama-compatible architecture (model_type "llama"), so the whole Llama tooling ecosystem works out of the box.
- Grouped-query attention even at 6B scale: 32 query heads over 4 KV heads, unusual for a model this small at the time.
- High rope_theta (5e6) chosen with long-context extension in mind; the Yi-6B-200K variant stretches the same architecture to 200k tokens.
- Compact 64000-token bilingual vocabulary.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

