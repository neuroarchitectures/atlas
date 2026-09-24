# Architecture: BERT-Base

## Motivation

The bidirectional encoder that started the transfer-learning era in NLP. Twelve post-norm encoder blocks, 768 hidden, 12 heads: the shape every "base-size" encoder since has copied.

## Core Idea

The bidirectional encoder that started the transfer-learning era in NLP.

## Architecture

### Overview

![BERT-Base architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 51 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder (BERT family) |
| Parameters | 110M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3,072, GeLU |
| Normalization | LayerNorm, post-norm |
| Positions | Absolute learned, max 512 |
| Vocabulary | 30,522 |

`model.json` is the full 12-layer graph (the repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face", with all hyperparameters from the official `config.json`.

### Design Notes

- The model that made pretrain-then-finetune the default workflow in NLP (Devlin et al. 2018, arXiv 1810.04805).
- Post-norm placement: LayerNorm comes after each residual add, the original Transformer ordering that pre-norm models later abandoned for training stability.
- Learned absolute position embeddings hard-cap the context at 512 tokens.
- 30522-token WordPiece vocabulary; masked-language-modeling plus next-sentence-prediction pretraining.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **108.5M**.
Hugging Face safetensors metadata reports **110.1M** for the real weights.
Deviation from the authoritative count (110.1M): **-1.5%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- The model that made pretrain-then-finetune the default workflow in NLP (Devlin et al. 2018, arXiv 1810.04805).
- Post-norm placement: LayerNorm comes after each residual add, the original Transformer ordering that pre-norm models later abandoned for training stability.
- Learned absolute position embeddings hard-cap the context at 512 tokens.
- 30522-token WordPiece vocabulary; masked-language-modeling plus next-sentence-prediction pretraining.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

