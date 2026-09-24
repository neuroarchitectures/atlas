# Rotary Position Embedding (RoPE)

## Design Philosophy

Encode position by *rotating* the Q and K vectors by an angle that depends on position, rather than adding a position vector. The philosophy: relative position falls out of the dot product (`<R_q q, R_k k> = <q, R_{k-q} k>`), so the model never needs an explicit position embedding, and it extrapolates far better than a learned table. The scheme used by Llama, Qwen, Mistral, and most modern decoders.

## Functionality

- For position `m` and dimension pair `(d_{2i}, d_{2i+1})`, rotate by angle `m · θ_i`, where `θ_i = base^{-2i/d}` (base = 10000).
- `RoPE(x, m) = x · cos(mθ) + rotate90(x) · sin(mθ)`.
- Applied to Q and K *inside* the attention op (a side input, not a main-path add).
- `dim` is the per-head rotation dimension; `maxSeqLen` sets the frequency base.
- **NTK / YaRN scaling**: Tweaking `base` or rescaling angles is how RoPE models extend context after training.

## Used By

| Model | Role |
|-------|------|
| LLaMA-2/3 | RoPE on Q/K, rotary dim = headDim |
| Mistral / Mixtral | Same |
| Qwen, Yi, Phi, Gemma | Variants |
| DeepSeek-V2 | Decoupled RoPE (separate from MLA latent) |

## Features

- **Relative position**: Encodes relative, not absolute, position — extrapolates past `maxSeqLen`.
- **No main-path add**: Position is a side input to attention, not an embedding sum.
- **Scalable**: NTK/YaRN scaling extends context post-training.

## Evolution

- **Predecessor**: Learned absolute positions (GPT, BERT); sinusoidal (original Transformer); ALiBi (linear bias on scores).
- **Successor**: Extensions like YaRN, LongRoPE for 1M+ context; decoupled RoPE in MLA.
