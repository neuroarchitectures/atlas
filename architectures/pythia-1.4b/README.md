# Pythia-1.4B

## Overview

One rung of EleutherAI's Pythia suite, the interpretability-and-training-dynamics workhorse: an identical GPT-NeoX architecture trained at 8 sizes on the exact same data order, with 154 checkpoints each. The 1.4B is the popular mid-size.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

One rung of EleutherAI's Pythia suite, the interpretability-and-training-dynamics workhorse: an identical GPT-NeoX architecture trained at 8 sizes on the exact same data order, with 154 checkpoints each.

## Key Characteristics

- GPT-NeoX parallel-residual block: attention and MLP both consume the same normed input and sum into the residual.
- Full RoPE on 128-dim heads; untied embeddings; 50304-token GPT-NeoX BPE vocabulary (padded for efficiency).
- The value is the controlled suite, not the architecture: same data, same order, every size and checkpoint released, so you can study how a fixed architecture learns.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 1.4B |
| Layers | 24 |
| Hidden size | 2048 |
| Attention | Multi-head: 16 heads, head dim 128 |
| Block | Parallel residual (GPT-NeoX) |
| FFN | Dense MLP, 8192, GeLU |
| Normalization | LayerNorm, pre-norm |
| Positions | RoPE (full) |
| Vocabulary | 50,304 |
| Max context | 2,048 |

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
