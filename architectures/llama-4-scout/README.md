# Llama 4 Scout

## Overview

Meta's first MoE and first 10M-context model. Architecturally the interesting bits are top-1 routing over 16 fat experts and iRoPE's position-free layers; reception was mixed, but the design choices are worth studying.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Meta's first MoE and first 10M-context model.

## Key Characteristics

- Coarse MoE, the opposite of DeepSeek's recipe: 16 large experts with top-1 routing plus a shared expert, in every layer (interleave step 1).
- iRoPE for the 10M-token context claim: interleaved layers drop positional encoding entirely (12 of 48 are NoPE, verified from config), betting that attention-only layers generalize past the trained length.
- GQA 40:8 at 5120 hidden; 202048-token vocabulary; natively multimodal (a 34-layer vision tower feeds the same decoder; this entry shows the text stack).
- Hyperparameters verified via the unsloth mirror of the config (the meta-llama repo is gated).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 109B total, 17B active |
| Layers | 48 |
| Hidden size | 5120 |
| Attention | Grouped-query: 40 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | MoE: 16 routed experts, top-1 + 1 shared, expert dim 8,192 |
| Normalization | RMSNorm, pre-norm |
| Positions | iRoPE: RoPE on 36 of 48 layers, every 4th layer position-free (NoPE) |
| Vocabulary | 202,048 |
| Max context | 10,485,760 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Llama 4 Community License. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
