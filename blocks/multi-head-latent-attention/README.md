# Multi-Head Latent Attention (MLA)

## Design Philosophy

Compress the KV cache into a low-dimensional **latent vector**. The philosophy: at inference, the KV cache dominates memory, and most of it is redundant. MLA projects K and V into a shared latent `c_KV` (e.g., 512-dim) that is cached instead of full K/V. At attention time, K and V are reconstructed from the latent. The decoupled RoPE is stored separately to preserve position information. The result: a KV cache 10–100× smaller than MHA, with no quality loss.

## Functionality

- **Down-project**: `c_KV = W_DKV · h` — compress the hidden state to a `d_c`-dim latent (e.g., 512).
- **Up-project K, V**: `K = W_UK · c_KV`, `V = W_UV · c_KV` — reconstruct per-head K, V.
- **Decoupled RoPE**: A separate small `q_rope, k_rope` path carries rotary position (since RoPE doesn't commute with the low-rank projection).
- **Cache**: Only `c_KV` (and the small RoPE K) is cached — `d_c` instead of `numHeads × headDim`.

## Used By

| Model | Role |
|-------|------|
| DeepSeek-V2 | Introduced MLA; 128 heads, KV latent 512 |
| DeepSeek-V2-Lite | The 16B runnable version |
| DeepSeek-V3 | Scaled to 671B, same MLA |
| DeepSeekMath | MLA for math reasoning |

## Features

- **Tiny KV cache**: ~10–100× smaller than MHA at the same head count.
- **No quality loss**: The latent is wide enough to reconstruct K/V faithfully.
- **Decoupled RoPE**: Position info is preserved via a side path.

## Evolution

- **Predecessor**: GQA / MQA (share KV heads, but still cache full K/V per head).
- **Successor**: Adopted across the DeepSeek line; an active research direction for further compression.
