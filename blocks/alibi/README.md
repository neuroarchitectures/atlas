# ALiBi (Attention with Linear Biases)

## Design Philosophy

No position embedding at all — instead, add a static, per-head **linear penalty** proportional to the distance between query and key to the attention scores. The philosophy: far-apart tokens are down-weighted, and because the penalty is computed at attention time (not a learned table), the trained context length is not a hard ceiling — ALiBi extrapolates cleanly to sequences longer than it ever saw.

## Functionality

- No `pos_embed` node in the main path.
- An `alibi` node adds a bias `b_{i,j} = -m_h · |i - j|` to the attention scores, where `m_h` is a per-head slope (geometric progression).
- The penalty is fixed (not learned), one slope per head.
- Softmax then proceeds normally.

## Used By

| Model | Role |
|-------|------|
| BLOOM | Position encoding |
| MPT | Position encoding |
| Various decoder LLMs | Extrapolation-focused variants |

## Features

- **Zero learned position params**: The bias is a fixed function of distance.
- **Clean extrapolation**: Trains short, runs long — ALiBi's headline property.
- **Per-head slopes**: Different heads decay at different rates.

## Evolution

- **Predecessor**: Learned absolute positions (don't extrapolate); sinusoidal (extrapolates but weakly).
- **Successor**: RoPE (relative, extrapolates, more expressive) became the dominant choice; ALiBi remains for extrapolation-critical use cases.
