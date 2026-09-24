# Native Sparse Attention Block

## Overview

The same pre-norm residual attention sub-block as [attn-full](../attn-full/), with one node swapped for **block-sparse attention**: the sequence is chunked into blocks and each query attends only to a selected top-k of them. Long-range reach without the full n×n cost, and hardware-aligned (block-granular) so it is fast in practice. The mechanism behind DeepSeek's Native Sparse Attention.

**Third of three sibling blocks** (full → sliding-window → sparse), identical except the attention op. See [COMPARISONS.md → Attention sparsity](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The same pre-norm residual attention sub-block as [attn-full](.

## Key Characteristics

- Two knobs: `blockSize` (granularity of a chunk) and `topBlocks` (how many chunks each query keeps). Unlike a fixed window, the selected blocks can be anywhere in the sequence, so it keeps genuine long-range links.
- Block granularity is the point: selection and compute align to hardware tiles, which is why it stays fast where token-level sparsity does not.
- Drop-in with full / sliding-window attention (same I/O shape), the contrast the [comparison](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse) draws.

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
