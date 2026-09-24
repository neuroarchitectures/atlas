# gpt-oss-120b

## Overview

The larger of OpenAI's two 2025 open-weight MoEs, sized to run on a single 80GB GPU at MXFP4. Same recipe as [gpt-oss-20b](../gpt-oss-20b/) but 128 experts and 36 layers: 117B total, 5.1B active.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The larger of OpenAI's two 2025 open-weight MoEs, sized to run on a single 80GB GPU at MXFP4.

## Key Characteristics

- 128 experts, top-4 routed, no shared expert; 5.1B of 117B parameters active per token.
- Alternating attention: odd layers a 128-token sliding window, even layers full attention, plus learned attention-sink logits.
- Ships MXFP4-quantized; trained with the harmony response format for tool use and chain-of-thought.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 117B total, 5.1B active |
| Layers | 36 |
| Hidden size | 2880 |
| Attention | GQA 64:8, head dim 64 |
| FFN | MoE: 128 experts, top-4 (no shared) |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE + YaRN; alternating sliding-window (128) and full attention |
| Vocabulary | 201,088 |
| Max context | 131,072 |

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
