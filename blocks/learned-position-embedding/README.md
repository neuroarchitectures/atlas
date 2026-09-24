# Learned Absolute Position Embedding

## Design Philosophy

A trainable vector per position, added to the token embeddings before attention. The philosophy: position is just another feature — learn it from data, like the token embeddings. Simple and effective, but capped at `maxLen` positions: it does not extrapolate past the trained length.

## Functionality

- A `maxLen × embedDim` embedding table.
- `pos_embed[pos]` is added to `token_embed[token_id]` → `x = embed + pos_embed`.
- Attention itself has no positional input; the position signal is entirely in the main-path add.

## Used By

| Model | Role |
|-------|------|
| GPT-2 / GPT-3 | Learned absolute positions |
| BERT | Learned absolute positions |
| ViT-B/16 | Learned positions (196 patches + CLS) |
| PatchTST | Learned positions on time-series patches |

## Features

- **Simple**: One embedding table, one add.
- **Trainable**: Learns the right position representation from data.
- **No extrapolation**: Capped at `maxLen` — the weakness that motivated RoPE and ALiBi.

## Evolution

- **Predecessor**: Sinusoidal positions (original Transformer) — fixed, extrapolates weakly.
- **Successor**: RoPE (relative, extrapolates); ALiBi (no embedding, extrapolates). Learned absolute is now mostly used in vision (ViT) and older LLMs.
