# Differential Attention

## Design Philosophy

Compute attention as the **difference of two separate softmax maps**: `Attn₁ - λ·Attn₂`. The philosophy: noise (attention to irrelevant context) is common to both maps, so subtraction cancels it — like noise-canceling headphones (differential signaling in EE). The result is a sparse attention pattern that amplifies signal and suppresses noise, addressing the "attention noise" that causes hallucination and retrieval failures.

## Functionality

Per head:
- Split Q, K, V into two halves: `Q₁, K₁, V₁` and `Q₂, K₂, V₂`.
- `Attn₁ = softmax(Q₁K₁^T/√d) V₁`
- `Attn₂ = softmax(Q₂K₂^T/√d) V₂`
- Output: `Attn₁ - λ·Attn₂`, where `λ` is a learnable scalar per head.
- Head dimension is halved to align FLOPs/params with standard MHA.

## Used By

| Model | Role |
|-------|------|
| Differential Transformer | The defining block, 3B model on 1T tokens |

## Features

- **Noise cancellation**: Subtraction removes attention mass common to both maps (irrelevant context).
- **Sparse patterns**: The difference is naturally sparser than either map.
- **Learnable λ**: Per-head control of cancellation strength.
- **FLOP-neutral**: Halved head dim keeps compute matched to standard MHA.

## Evolution

- **Predecessor**: Standard multi-head attention; differential signaling (EE).
- **Successor**: Potential integration with sparse/linear attention variants; scaling to larger models.
