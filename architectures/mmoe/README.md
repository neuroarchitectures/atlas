# MMoE

## Overview

Multi-gate Mixture-of-Experts: instead of one shared bottom feeding every task (which forces conflicting objectives to share a representation), MMoE gives each task a softmax gate that mixes a shared pool of experts its own way. The default multi-task ranker behind feed and video systems that optimize several objectives at once.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Multi-gate Mixture-of-Experts: instead of one shared bottom feeding every task (which forces conflicting objectives to share a representation), MMoE gives each task a softmax gate that mixes a shared pool of experts its own way.

## Key Characteristics

- Every task reads the same experts but through its own gate, so tasks that disagree can lean on different experts without a hard split.
- Reference topology: the per-task gate softmax is shown joining the expert mixture; in the paper the gate weights are the mixture coefficients.
- Compare [ple](../ple/), which hardens the soft gating into explicit task-specific vs shared expert groups.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Multi-task ranking |
| Experts | Shared bottom feeds N expert MLPs |
| Gates | Per-task softmax gate over the experts |
| Towers | One MLP head per objective |
| Key idea | Soft expert sharing avoids multi-task negative transfer |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

## Related Architectures

See `references/README.md` for related architectures and research context.
