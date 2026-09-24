# Full Attention Block

## Overview

A minimal pre-norm residual attention sub-block with **dense (full) self-attention**: every token attends to every other token. The O(n²) baseline that the sliding-window and sparse variants trade away for longer context.

This is the **first of three sibling blocks** that are identical except for the attention op, so the diff is exactly the attention mechanism. See [COMPARISONS.md → Attention sparsity](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A minimal pre-norm residual attention sub-block with **dense (full) self-attention**: every token attends to every other token.

## Key Characteristics

- Dense attention: full n×n score matrix, quadratic in sequence length. Maximum expressivity, maximum cost.
- Pre-norm placement (norm before attention, residual around it), the modern default.
- Fork point: swap node 3 for `localAttention` or `nativeSparseAttention` to see the [sibling blocks](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse), and re-validate instantly.

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
