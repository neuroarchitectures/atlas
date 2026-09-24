# Skywork-13B

## Overview

Kunlun Tech's 13B bilingual base model, notable for an explicitly ablated deep-and-thin layout (52 layers) and a thorough public tech report on its 3.2T-token training run.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Kunlun Tech's 13B bilingual base model, notable for an explicitly ablated deep-and-thin layout (52 layers) and a thorough public tech report on its 3.

## Key Characteristics

- Deliberately deep-and-thin at 13B scale: 52 layers at 4608 hidden, versus Llama-2-13B's 40 layers at 5120. The tech report (arXiv 2310.19341) ablates this choice directly.
- Plain multi-head attention (36 heads), RoPE, RMSNorm, SwiGLU; the FFN ratio is a modest 2.67x.
- 65519-token vocabulary balanced for Chinese and English; trained on 3.2T tokens.
- Ships with a clean tech report and intermediate checkpoints, making it a good reference for training-dynamics work.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 13.8B |
| Layers | 52 |
| Hidden size | 4608 |
| Attention | Multi-head: 36 heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 12,288 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 65,519 |
| Max context | 4,096 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code Apache-style; weights under the Skywork Community License (free commercial use after agreement). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
