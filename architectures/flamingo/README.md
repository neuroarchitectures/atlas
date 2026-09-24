# Flamingo

## Overview

The visual-language model that handles interleaved images and text and does few-shot in-context learning. A Perceiver Resampler compresses image features to a few tokens, and gated cross-attention layers spliced into a frozen LLM let the text attend to them.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The visual-language model that handles interleaved images and text and does few-shot in-context learning.

## Key Characteristics

- Perceiver Resampler: a fixed set of learned latents cross-attend the variable-length vision features, so any image (or video) becomes a constant 64 visual tokens.
- tanh gating: the inserted cross-attention starts at zero contribution (gate = 0), so the model begins as exactly the frozen LLM and learns to use vision gradually, which is what made training stable.
- The ancestor of the "freeze the LLM, splice in vision via cross-attention" branch of MLLMs; contrast with the prefix-token approach of [llava-1.5-7b](../llava-1.5-7b/).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Few-shot visual language model |
| Vision encoder | Frozen (NFNet / CLIP) |
| Resampler | Perceiver: learned latents cross-attend vision features → fixed tokens |
| LLM | Frozen, with gated cross-attention layers inserted between blocks |
| Key idea | Interleave image + text, tanh-gated x-attn so init = pure LLM |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Research (weights not released; open re-impl: OpenFlamingo). The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
