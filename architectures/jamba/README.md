# Jamba

## Overview

AI21's 2024 hybrid: the first production-scale model to interleave Mamba state-space mixers with Transformer attention, plus MoE. Most layers are Mamba (linear-time, no KV cache); one in eight is attention (for in-context recall); MoE replaces every other MLP. The result fits a 256K context on a single 80GB GPU.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

AI21's 2024 hybrid: the first production-scale model to interleave Mamba state-space mixers with Transformer attention, plus MoE.

## Key Characteristics

- Hybrid mixer stack: 7 Mamba SSM layers per 1 attention layer, so the KV cache and quadratic cost only appear on 1/8 of layers.
- MoE every other layer: 16 experts, top-2 routing, giving 52B total but ~12B active per token.
- Shown as a structural reference: the SSM + MoE parameter mix is documented (52B / 12B) rather than recomputed by the per-layer estimator, so this entry carries no param-gate deviation.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Hybrid SSM-Transformer-MoE decoder (causal LM) |
| Parameters | 52B total / 12B active |
| Layers | 32 (4 blocks of 8) |
| Hidden size | 4,096 |
| Mixers | 7 Mamba (SSM) + 1 attention per 8-layer block |
| Attention | GQA: 32 query heads, 8 KV heads (1 in every 8 layers) |
| FFN | MoE on odd layers (16 experts, top-2); single MLP on even layers |
| Normalization | RMSNorm, pre-norm |
| Positions | None; Mamba mixers carry order through the SSM recurrence |
| Max context | 256K |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
