# T5-Small

## Overview

The text-to-text encoder-decoder that reframed every NLP task as sequence generation. The full two-stream graph: bidirectional encoder, causal decoder, and cross-attention tying them together.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The text-to-text encoder-decoder that reframed every NLP task as sequence generation.

## Key Characteristics

- Both streams and the LM head share one 32128-token SentencePiece embedding matrix (see the parameter note below).
- RMSNorm (T5 called it "simplified LayerNorm") years before the Llama lineage made it standard; relative position biases instead of absolute embeddings.
- The graph makes the three attention types visually distinct: encoder self-attention, decoder causal self-attention, and cross-attention.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Encoder-decoder transformer (text-to-text) |
| Parameters | 60.5M |
| Layers | 6 encoder + 6 decoder |
| Hidden size | 512 |
| Attention | 8 heads; decoder adds cross-attention |
| FFN | Dense, 2048, ReLU |
| Normalization | RMSNorm, pre-norm |
| Positions | Relative position biases |
| Vocabulary | 32,128 (shared) |

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
