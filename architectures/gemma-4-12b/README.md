# Gemma 4 12B

## Overview

The 12B of Google's Gemma 4 generation, the strongest permissively-distributed dense model of the June 2026 wave. Wide 256-dim heads, 5:1 local:global attention, and a 262K vocabulary.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The 12B of Google's Gemma 4 generation, the strongest permissively-distributed dense model of the June 2026 wave.

## Key Characteristics

- Hyperparameters read directly from the June 2026 config.json (gemma4_unified_text): this entry tracks the release, not secondhand writeups.
- Oversized heads: 16 heads of dim 256 (4096-dim attention over 3840 hidden), continuing the Gemma trademark of few-but-wide heads.
- 5:1 local-to-global attention with a 1024-token window keeps the 262144-token context affordable; only 8 of 48 layers see the full sequence.
- GeGLU FFN at 15360 (4x hidden) and the huge 262144-token SentencePiece vocabulary; "unified" model_type with a built-in vision encoder, and the imported graph includes the vision patch-embed entry path feeding the shared decoder.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 12B |
| Layers | 48 |
| Hidden size | 3840 |
| Attention | Grouped-query: 16 query heads, 8 KV heads |
| Head dim | 256 |
| FFN | GeGLU, hidden size 15,360 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE; 5:1 local(1024-window):global attention layers (40 sliding, 8 full, verified from config) |
| Vocabulary | 262,144 |
| Max context | 262,144 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Gemma Terms of Use. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
