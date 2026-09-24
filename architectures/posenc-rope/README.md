# Rotary Positions (RoPE)

## Overview

The same minimal decoder block as [posenc-learned](../posenc-learned/), but position is encoded by **rotary embeddings (RoPE)**: instead of adding a position vector, the query/key vectors are *rotated* by an angle that depends on position, inside the attention op. Relative position falls out of the dot product, and it extrapolates far better than a learned table. The scheme used by Llama, Qwen, Mistral, and most modern decoders.

**Second of three sibling blocks** (learned → RoPE → ALiBi). Here the position signal is a `rope` node wired **into the attention** as a side input, not into the residual stream. See [COMPARISONS.md → Positional encoding](../../COMPARISONS.md#positional-encoding-learned--rope--alibi).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The same minimal decoder block as [posenc-learned](.

## Key Characteristics

- No vector is added to the embeddings: the `rope` node rotates Q/K **inside attention**, so the position signal is a side input to node 5, not a main-path add.
- `dim` is the per-head rotation dimension (here headDim = 512 / 8 = 64); `maxSeqLen` sets the frequency base. Tweaking these (NTK / YaRN scaling) is how RoPE models extend context after training.
- Encodes *relative* position, which is why it extrapolates past `maxSeqLen` far better than [learned absolute](../posenc-learned/) positions.

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
