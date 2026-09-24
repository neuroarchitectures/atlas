# PLE (CGC)

## Overview

Progressive Layered Extraction (and its single-level form CGC): splits experts into task-specific groups and a shared group, so each task draws on its own specialists plus the shared pool but never on another task's specialists. Tencent's answer to the "seesaw" effect where improving one task quietly degrades another.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Progressive Layered Extraction (and its single-level form CGC): splits experts into task-specific groups and a shared group, so each task draws on its own specialists plus the shared pool but never on another task's specialists.

## Key Characteristics

- CGC is the single-extraction-layer version; PLE stacks several extraction layers to progressively separate and recombine representations.
- Reference topology: one extraction layer with a task-A expert, a shared expert, and a task-B expert.
- Direct lineage from [mmoe](../mmoe/): same multi-task goal, but explicit expert grouping instead of purely soft gates.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Multi-task ranking |
| Experts | Task-specific groups plus a shared group |
| Routing | Each task uses its own experts plus shared |
| Towers | One MLP head per objective |
| Key idea | Explicit expert separation cuts negative transfer |

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
