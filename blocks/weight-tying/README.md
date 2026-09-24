# Weight Tying

## Design Philosophy

Input embedding and output unembedding are the same transformation in reverse — so share the matrix. Tying saves V×d parameters, regularizes the embedding space (tokens are pushed toward useful output geometry), and notably helps small models.

## Functionality

- `logits = h · E^T` with E the input embedding matrix; no separate LM-head weight (optionally separate bias / norm, e.g., Baichuan's normalized head variant).

## Used By

| Model | Role |
|-------|------|
| GPT-1 | 12-layer decoder, W_e tied to LM head |
| MiniCPM-2B | Tied embeddings + muP-style scaled residuals (scale_emb=12) |

## Features

- **Parameter savings** at vocab scale — significant for small/medium models.
- **Implicit regularization** — embedding geometry serves both directions.

## Evolution

- **Predecessor**: Press & Wolf 2017 (tying recommendation); ALBERT factorized embeddings.
- **Related**: un-tied heads (GPT-2/3, LLaMA) dominate at scale — the benefit fades with capacity; normhead variants normalize head rows for stability.
