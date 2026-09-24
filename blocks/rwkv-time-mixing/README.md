# RWKV Time-Mixing Block

## Design Philosophy

A linear attention mechanism that can be computed as either a Transformer (parallel training) or an RNN (constant-memory inference). The philosophy: replace the pairwise T×T attention matrix with a **channel-wise time-decay vector** W, reducing space from O(T²) to O(d). The model is exact (not an approximation of attention), and the dual formulation gives parallel training + O(d) inference.

## Functionality

- **Token shift**: Linear interpolation between current and previous input.
- **Time-mixing**: R, K, V computed from shifted input. The WKV state is updated:
  `WKV_t = (Σ_{i=1}^{t} e^{-(t-i)w + k_i} v_i + e^{u_t + k_t} v_t) / (Σ e^{-(t-i)w + k_i} + e^{u_t + k_t})`
  where `w ∈ R≥0^d` is a per-channel decay vector, `u` compensates for current-token degeneration.
- **Receptance (R)**: `σ(...)` gates the WKV output.
- **Channel-mixing**: A separate gated FFN-like block.
- Both wrapped in residual + LayerNorm.

## Used By

| Model | Role |
|-------|------|
| RWKV (v1–v6) | The defining architecture |
| Various RWKV-based LLMs | 169M to 14B |

## Features

- **O(Td) training, O(d) inference**: Linear in sequence length, constant memory at inference.
- **Dual mode**: Parallel (Transformer) for training, sequential (RNN) for inference.
- **Exact**: Not an approximation of attention.
- **Channel-wise decay**: W controls forgetting per channel.

## Evolution

- **Predecessor**: LSTM (not parallelizable); Transformer (quadratic); AFT (position biases).
- **Successor**: RWKV-v4/5/6; Mamba (input-dependent, vs RWKV's fixed decay).
