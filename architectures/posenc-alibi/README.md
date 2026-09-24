# Linear Bias Positions (ALiBi)

## Overview

The same minimal decoder block as [posenc-learned](../posenc-learned/), but position is encoded by **ALiBi (Attention with Linear Biases)**: no position embedding at all. A static, per-head linear penalty proportional to the distance between query and key is added to the attention scores, so far-apart tokens are down-weighted. Trains short, extrapolates long. The scheme used by MPT and BLOOM.

**Third of three sibling blocks** (learned → RoPE → ALiBi). Like RoPE, the position signal (`alibi`) feeds **into the attention**, not the residual stream, but it is a bias on scores rather than a rotation of Q/K. See [COMPARISONS.md → Positional encoding](../../COMPARISONS.md#positional-encoding-learned--rope--alibi).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The same minimal decoder block as [posenc-learned](.

## Key Characteristics

- Zero learned position parameters: the `alibi` node adds a fixed distance penalty to attention scores, one slope per head (hence `numHeads`).
- The penalty is computed at attention time, so the trained context length is not a hard ceiling, ALiBi's headline property is clean extrapolation to sequences longer than it ever saw.
- Same wiring slot as [posenc-rope](../posenc-rope/) (side input to attention), different mechanism: a bias on scores vs. a rotation of Q/K. The [comparison](../../COMPARISONS.md#positional-encoding-learned--rope--alibi) puts the three side by side.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

## Related Architectures

See `references/README.md` for related architectures and research context.
