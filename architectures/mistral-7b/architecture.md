# Architecture: Mistral-7B

## Motivation

The 7B model from Mistral AI that beat Llama-2-13B at release, built on grouped-query attention plus sliding-window attention. The direct dense ancestor of Mixtral.

## Core Idea

The 7B model from Mistral AI that beat Llama-2-13B at release, built on grouped-query attention plus sliding-window attention.

## Architecture

### Overview

![Mistral-7B architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 197 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.2B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 32,000 |
| Max context | 32,768 |

`model.json` is the full 32-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- Sliding-window attention: each layer attends to the previous 4096 tokens only; stacking 32 layers grows the effective receptive field to the full 32768-token context. The graph shows standard GQA; the window is an attention-mask detail.
- Grouped-query attention 32:8 plus the sliding window made it the fastest-inference 7B of its generation.
- Kept the compact Llama-2 32000-token vocabulary while adopting the larger 14336 FFN.
- The dense base that Mixtral 8x7B sparsified: the MoE model reuses this exact block with the FFN swapped for 8 experts.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **7.24B**.
Hugging Face safetensors metadata reports **7.24B** for the real weights.
Deviation from the authoritative count (7.24B): **+0.0%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Sliding-window attention: each layer attends to the previous 4096 tokens only; stacking 32 layers grows the effective receptive field to the full 32768-token context. The graph shows standard GQA; the window is an attention-mask detail.
- Grouped-query attention 32:8 plus the sliding window made it the fastest-inference 7B of its generation.
- Kept the compact Llama-2 32000-token vocabulary while adopting the larger 14336 FFN.
- The dense base that Mixtral 8x7B sparsified: the MoE model reuses this exact block with the FFN swapped for 8 experts.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

