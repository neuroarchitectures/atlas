# Learned Absolute Positions

## Overview

A minimal decoder attention block where position information comes from a **learned absolute position embedding** added to the token embeddings. One trainable vector per position, summed into the input before attention. The scheme used by the original GPT and BERT.

**First of three sibling blocks** showing how position is injected: learned → RoPE → ALiBi. The graph diff is the whole point, here the position signal is a visible node in the main path (`pos_embed`) and attention itself is plain. See [COMPARISONS.md → Positional encoding](../../COMPARISONS.md#positional-encoding-learned--rope--alibi).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A minimal decoder attention block where position information comes from a **learned absolute position embedding** added to the token embeddings.

## Key Characteristics

- Position lives in the **main path**: a `maxLen × embedDim` table added to the token embeddings, so attention needs no positional input.
- Trainable and simple, but capped at `maxLen` positions, this is the variant that does not extrapolate past its trained length, which is what motivated RoPE and ALiBi.
- Contrast with the [siblings](../../COMPARISONS.md#positional-encoding-learned--rope--alibi): in [posenc-rope](../posenc-rope/) and [posenc-alibi](../posenc-alibi/) the position node feeds the attention op instead of the residual stream.

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
