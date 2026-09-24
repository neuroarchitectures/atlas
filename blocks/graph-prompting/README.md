# Graph Prompt Tokens

## Design Philosophy

Pretrain once, prompt for every target domain — GNN edition. Learnable *structure tokens* injected into GNN aggregation harmonize structural distributions across source domains, while *dual prompts* (holistic cross-domain + domain-specific) are concatenated into node/graph representations for lightweight adaptation — no backbone fine-tuning.

## Functionality

- Structure tokens added at each aggregation step during multi-domain pre-training.
- Target-domain adaptation: prompt = concat(holistic prompt, specific prompt) into node/graph embeddings before the head.

## Used By

| Model | Role |
|-------|------|
| SAMGPT | Multi-domain graph pre-training + prompt-based adaptation over GCN backbones |

## Features

- **Distribution harmonization** — structure tokens normalize cross-domain graph statistics.
- **Prompt-only adaptation** — the transfer analogue of visual prompt tuning.

## Evolution

- **Predecessor**: GraphPrompt / GPPT prompt learning on graphs.
- **Related**: "prompt" mechanisms in SAM's prompt-encoder — the same prefix-conditioning pattern.
