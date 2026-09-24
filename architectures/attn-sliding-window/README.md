# Sliding-Window Attention Block

## Overview

The same pre-norm residual attention sub-block as [attn-full](../attn-full/), with one node swapped: each token attends only to a fixed **window** of nearby tokens instead of the whole sequence. Cost drops from O(n²) to O(n·w), and stacking layers still grows the effective receptive field. The mechanism behind Longformer and Mistral.

**Second of three sibling blocks** (full → sliding-window → sparse), identical except the attention op. See [COMPARISONS.md → Attention sparsity](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The same pre-norm residual attention sub-block as [attn-full](.

## Key Characteristics

- `windowSize` is the one knob: each query sees `windowSize` neighbours, so attention is linear in sequence length.
- Receptive field grows with depth: L stacked windows of size w reach ~L·w tokens, the trick that lets Mistral run a 4K window over much longer context.
- Same I/O shape as full attention, so it is a drop-in swap, exactly what the [comparison](../../COMPARISONS.md#attention-sparsity-full--sliding-window--sparse) shows.

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
