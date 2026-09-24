# Multi-Head Attention (Full / Dense)

## Design Philosophy

Project Q, K, V into `h` different subspaces, run scaled dot-product attention in each in parallel, then concatenate and project. The philosophy: a single attention head averages positions and loses resolution; multiple heads let the model *jointly* attend to information from different representation subspaces at different positions — like multiple channels in a CNN.

## Functionality

Per head `i`: `head_i = softmax(Q_i K_i^T / √d_k) V_i`, where `Q_i = xW_i^Q`, etc., and `d_k = d_model / h`.
Output: `MultiHead = Concat(head_1, ..., head_h) W^O`.

- **Scaling**: `1/√d_k` counteracts the variance growth of dot products in high dimensions (prevents softmax saturation).
- **Dense (full) attention**: Every query attends to every key — the n×n score matrix, O(n²) cost.
- **Causal mask** (decoder): Additive mask zeroing future positions.

## Used By

| Model | Role |
|-------|------|
| Transformer (original) | 8 heads, d_k=64 |
| BERT | 12 heads |
| GPT-2/3 | 12/96 heads |
| ViT | 12 heads |

## Features

- **Maximum expressivity**: Full n×n attention, every token sees every other.
- **Multi-subspace**: Different heads learn different relations (syntactic, coreference, etc.).
- **O(n²) cost**: The baseline that sparse / linear variants trade away.

## Evolution

- **Predecessor**: Additive attention (Bahdanau); self-attention used alongside RNNs.
- **Successor**: Sparse attention (Longformer, Big Bird); GQA/MQA (share KV); FlashAttention (exact, faster).
