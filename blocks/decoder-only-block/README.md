# Decoder-Only Block (GPT/LLaMA style)

## Design Philosophy

Drop the encoder-decoder cross-attention; keep only **masked self-attention + FFN**, both wrapped in residuals. The philosophy: a single stack of identical blocks, trained autoregressively, is enough — the model learns to predict the next token, and everything (comprehension, generation, in-context learning) emerges from that objective. Modern variants (LLaMA) swap LayerNorm→RMSNorm, ReLU-FFN→SwiGLU, and add RoPE.

## Functionality

Modern (LLaMA-style, pre-norm): `x → [RMSNorm → GQA(with RoPE) → add] → [RMSNorm → SwiGLU → add] → output`.

- **RMSNorm**: LayerNorm without the mean subtraction; cheaper, same effect.
- **Grouped-Query Attention (GQA)**: Multiple query heads share fewer KV heads — smaller KV cache.
- **RoPE**: Rotary position encoding applied to Q/K inside attention.
- **SwiGLU FFN**: `Swish(xW_G) ⊙ (xW_1) W_2` — gated FFN, better than ReLU-FFN.
- Two residual streams (one per sub-layer).

## Used By

| Model | Role |
|-------|------|
| GPT-2 / GPT-3 | Decoder-only, LayerNorm + ReLU FFN |
| LLaMA-3 | RMSNorm + GQA + RoPE + SwiGLU |
| Mistral-7B | Same, with sliding-window attention |
| Qwen, Yi, Phi, Gemma | Variants on the same recipe |

## Features

- **Uniform stack**: N identical blocks — the simplest scalable architecture.
- **Autoregressive**: Causal mask + KV cache for efficient generation.
- **Modern recipe**: RMSNorm + RoPE + SwiGLU + GQA is the open-source default.

## Evolution

- **Predecessor**: Transformer decoder (drops cross-attention → decoder-only).
- **Successor**: MoE decoder (Mixtral, DeepSeek) swaps the FFN for a routed expert mixture.
