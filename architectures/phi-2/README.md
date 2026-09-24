# Phi-2

## Overview

Microsoft's 2.7B "small model, textbook-quality data" flagship. Architecturally a GPT-NeoX-style parallel-residual decoder with partial rotary embeddings, included as the canonical example of the Phi data-over-scale thesis.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Microsoft's 2.

## Key Characteristics

- Parallel residual: attention and the MLP both read the same pre-norm input and their outputs are summed into the residual, so the two run side by side instead of in series (saves a norm, slightly faster).
- Partial RoPE: rotary embeddings are applied to only 40% of each 80-dim head; the rest is position-free, a GPT-NeoX inheritance.
- Plain dense GeLU MLP at 4x hidden, no GQA, no SwiGLU: the architecture is deliberately conventional. The Phi result is about data curation, not architecture.
- Untied input/output embeddings.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 2.78B |
| Layers | 32 |
| Hidden size | 2560 |
| Attention | Multi-head: 32 heads, head dim 80 |
| Block | Parallel residual (attention and MLP share one input) |
| FFN | Dense MLP, 10240, GeLU |
| Normalization | LayerNorm, pre-norm |
| Positions | Partial RoPE (40% of each head) |
| Vocabulary | 51,200 |
| Max context | 2,048 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** MIT. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
