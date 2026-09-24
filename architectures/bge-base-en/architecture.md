# Architecture: BGE-base-en-v1.5

## Motivation

BAAI's BGE retriever, for a long stretch the top open English embedding model on the MTEB leaderboard. A BERT-base encoder fine-tuned with large-scale contrastive pretraining + instruction tuning; the CLS token is the embedding.

## Core Idea

BAAI's BGE retriever, for a long stretch the top open English embedding model on the MTEB leaderboard.

## Architecture

### Overview

![BGE-base-en-v1.5 architecture](assets/diagram.png)

*Identical repeated blocks are folded into one representative block with a `× N` badge, so the whole architecture fits on screen. `model.json` uses a NAXS block template + `repeat` directive: the identical repeated layers are defined once and expand to all 51 nodes (open it in Neurarch to see and edit every layer). Vector: [diagram.svg](assets/diagram.svg).*

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder, retrieval embedding |
| Parameters | 109M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3072, GeLU |
| Normalization | LayerNorm, post-norm |
| Pooling | CLS token → 768-dim embedding |
| Vocabulary | 30,522 |
| Max context | 512 |

`model.json` is the full graph (repeated layers expressed via a NAXS block template + `repeat`), produced with the same import path the Neurarch app uses for "load from Hugging Face".

### Design Notes

- Architecturally plain BERT-base; all the lift is the C-Pack training recipe (contrastive pretraining on curated pairs, then task fine-tuning).
- Uses the CLS token as the sentence embedding (unlike all-MiniLM's mean pool), and recommends an instruction prefix for queries.
- The "base" tier balances quality and cost; -small and -large siblings trade the two.

### Parameter Check

Neurarch's per-layer parameter estimate over this graph: **108.5M**.
Hugging Face safetensors metadata reports **109.5M** for the real weights.
Deviation from the authoritative count (109.5M): **-0.9%**.

## Design Decisions

> **Analysis** — Observations derived from studying the architecture.

- Architecturally plain BERT-base; all the lift is the C-Pack training recipe (contrastive pretraining on curated pairs, then task fine-tuning).
- Uses the CLS token as the sentence embedding (unlike all-MiniLM's mean pool), and recommends an instruction prefix for queries.
- The "base" tier balances quality and cost; -small and -large siblings trade the two.

## Implementation Notes

- `model.json` contains the full structural graph with real dimensions and parameters, using a NAXS block template + `repeat` for the identical repeated layers.
- Open in [Neurarch](https://www.neurarch.com/) to visualize, edit, or export training code.
- Diagrams fold repeated identical blocks with a `× N` badge; `model.json` defines each repeated layer once at real dimensions and expands to all nodes.

## Limitations

- This package focuses on the architectural structure. Training pipelines, 
dataset preparation, and production optimizations are out of scope.

## Evolution

> See `references/related/` for predecessor and successor architectures.

