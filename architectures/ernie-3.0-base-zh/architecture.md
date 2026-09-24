# Architecture: ERNIE 3.0 Base (Chinese)

## Motivation

Baidu's workhorse Chinese encoder, the base-size distillation of the ERNIE 3.0 family. BERT-base shape with a larger vocabulary, 2048-token positions, and knowledge-enhanced pretraining.

## Core Idea

Baidu's workhorse Chinese encoder, the base-size distillation of the ERNIE 3.

## Architecture

### Overview

![ERNIE 3.0 Base (Chinese) architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 51 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder (BERT family) |
| Parameters | 118M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3,072, GeLU |
| Normalization | LayerNorm, post-norm |
| Positions | Absolute learned, max 2,048 |
| Vocabulary | 40,000 |

`model.json` is the full 12-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- BERT-base shape with two ERNIE twists: a 40000-token vocabulary (almost 2x BERT's Chinese vocab) and a 2048 max position, four times BERT's 512.
- Adds a task-type embedding alongside token and position embeddings, a remnant of ERNIE 3.0's multi-task universal-representation pretraining.
- Distilled from the 10B ERNIE 3.0 Titan teacher; the base model is what ships for practical NLU.
- Official weights are Paddle-native; the linked HF checkpoint is the standard PyTorch conversion by nghuyong.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **115.8M**.
Deviation from the authoritative count (118.0M): **-1.9%**.

> The graph sum lands ~2% under the official figure, which includes ERNIE's task-type embedding table that the importer does not model.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- BERT-base shape with two ERNIE twists: a 40000-token vocabulary (almost 2x BERT's Chinese vocab) and a 2048 max position, four times BERT's 512.
- Adds a task-type embedding alongside token and position embeddings, a remnant of ERNIE 3.0's multi-task universal-representation pretraining.
- Distilled from the 10B ERNIE 3.0 Titan teacher; the base model is what ships for practical NLU.
- Official weights are Paddle-native; the linked HF checkpoint is the standard PyTorch conversion by nghuyong.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

