# GLM-4.5-Air

## Overview

Zhipu AI's agent-focused MoE in its deployable Air size. Distinctive for spending parameters on attention (96 heads, 3x hidden) while keeping experts slim, the opposite allocation from most 2025 MoEs.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Zhipu AI's agent-focused MoE in its deployable Air size.

## Key Characteristics

- Wide attention: 96 query heads of dim 128 give a 12288-dim attention space over a 4096 hidden size (3x), an unusually attention-heavy budget the GLM-4.5 report credits for reasoning performance.
- GQA 96:8 with partial RoPE (half of each head), plus QKV bias (attention_bias = true), a Qwen2-style touch the rest of the 2025 wave dropped.
- 128 fine-grained experts, top-8 routing, 1 shared expert, slim 1408-dim experts; first layer dense at 10944.
- The "Air" tier of the GLM-4.5 agentic line: 106B total but only 12B active, sized to run on a single high-end node.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 106B total, 12B active |
| Layers | 46 |
| Hidden size | 4096 |
| Attention | Grouped-query: 96 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | MoE: 128 routed experts, top-8 + 1 shared, expert dim 1,408; first 1 layer dense (10,944) |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
| Vocabulary | 151,552 |
| Max context | 131,072 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** MIT. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
