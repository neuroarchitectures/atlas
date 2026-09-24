# YaRN RoPE Scaling

## Design Philosophy

RoPE-trained models degrade beyond their training context. YaRN extends context by interpolating RoPE frequencies *non-uniformly* — high frequencies (local order) are preserved, low frequencies (global position) are stretched — plus a temperature on the attention logits to compensate.

## Functionality

- Per-frequency interpolation factor blending NTK-aware scaling for low frequencies with linear interpolation for high ones; attention temperature `t ≈ 0.1 ln(k) + 1`.
- Applied post-training (no retraining) or with a short continued-pretraining ramp.

## Used By

| Model | Role |
|-------|------|
| gpt-oss-120b / gpt-oss-20b | 128K context on top of RoPE |

## Features

- **Frequency-selective stretching** — preserves short-range resolution that linear PI destroys.
- **Cheap context extension** — minutes of tuning, no architecture change.

## Evolution

- **Predecessor**: Position Interpolation (PI), NTK-aware scaling.
- **Successor**: interleaved-nope (avoid the length prior altogether in some layers); LongRoPE searched scaling.
