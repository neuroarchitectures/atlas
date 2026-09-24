# Transformer Encoder Block

## Design Philosophy

The block that replaced recurrence with attention. Two sub-layers — multi-head self-attention and a position-wise feed-forward network — each wrapped in a residual connection and layer normalization. The philosophy: every position attends to every other position in parallel, so long-range dependencies have path length O(1), and the whole sequence is processed in parallel (no sequential RNN bottleneck).

## Functionality

Structure (post-norm original; pre-norm modern): `input → [MHA → add → LayerNorm] → [FFN → add → LayerNorm] → output`.

- **Multi-Head Self-Attention**: `Q, K, V = linear(x)`; `softmax(QK^T/√d_k)V` per head, h heads, concat → linear.
- **FFN**: Two linear layers with ReLU (original) or GeLU (modern): `FFN(x) = GeLU(xW_1)W_2`, inner dim 4×.
- **Residual + LayerNorm**: Around each sub-layer.
- **Pre-norm** (modern default): `x + Sublayer(LayerNorm(x))` — more stable for deep stacks.

## Used By

| Model | Role |
|-------|------|
| BERT-Base | 12 encoder blocks, 768-dim, 12 heads |
| ViT-B/16 | 12 encoder blocks (pre-norm) |
| T5-Small (encoder) | Encoder blocks |
| CLIP (both towers) | 12 blocks each |
| Whisper (encoder) | 12 blocks |

## Features

- **Parallelism**: O(1) sequential ops, fully parallelizable across sequence length.
- **Global receptive field**: Every token sees every other token in one layer.
- **Modularity**: Stack N identical blocks; depth is the main scaling axis.

## Evolution

- **Predecessor**: RNN+attention (Bahdanau) — attention was an *add-on* to recurrence; the Transformer makes attention the *only* mechanism.
- **Successor**: Decoder block (adds masked self-attention + cross-attention); LLaMA-style decoder (RMSNorm + RoPE + SwiGLU).
