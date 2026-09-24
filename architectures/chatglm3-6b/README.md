# ChatGLM3-6B

## Overview

The third generation of the ChatGLM series from Zhipu AI and Tsinghua, historically the most-starred Chinese open LLM. A pre-norm decoder with extreme grouped (near-multi-query) attention and GLM-style partial rotary embeddings.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The third generation of the ChatGLM series from Zhipu AI and Tsinghua, historically the most-starred Chinese open LLM.

## Key Characteristics

- Aggressive multi-query-style attention: 32 query heads share only 2 KV groups (multi_query_group_num = 2), a 16:1 ratio that shrinks the KV cache far more than typical GQA.
- GLM-style partial RoPE: rotary embedding is applied to half of each 128-dim head (rotary dim 64); the other half stays position-free.
- The FFN intermediate size (13696) follows the GLM recipe rather than the Llama 8/3 ratio.
- 8192-token base context; -32k and -128k long-context variants exist. Input and output embeddings are untied (two 65024 x 4096 matrices).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.2B |
| Layers | 28 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 2 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 13,696 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
| Vocabulary | 65,024 |
| Max context | 8,192 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code Apache 2.0; weights under the ChatGLM3-6B license (free commercial use after registration). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
