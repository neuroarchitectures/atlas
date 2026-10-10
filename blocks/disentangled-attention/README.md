# Disentangled Attention (DeBERTa)

## Design Philosophy

In standard self-attention, a token's content and its position are added into a single vector, so the model cannot tell whether a strong attention score came from matching words or from matching nearby positions. DeBERTa disentangles the two and computes separate content-to-content, content-to-position, and position-to-content attention terms.

## Functionality

Represent each token by two vectors: content `{c_i}` and relative position `{p_{i,j}}`. The attention score is the sum of three disentangled components:

```
A_{ij} = c_i^T c_j                 (content -> content)
       + c_i^T p_{i,j}             (content -> position)
       + c_j^T p_{i,j}             (position -> content)
```

Each term has its own projection; the relative-position embedding is shared across heads and computed for the offset `j - i`.

## Used By

DeBERTa

## Features

- **Position/content separated**: no conflation of "what" and "where".
- **Relative positions**: generalises better than absolute embeddings.
- **Three attention terms**: content-to-position and position-to-content are asymmetric.
- **Shared position embedding**: cuts parameters versus per-head relative bias.

## Evolution

Predecessor: relative position bias / RoPE (position fused into the score). Sibling: relative-position bias, RoPE. Successor: DeBERTa-v2/v3 disentangled attention.
