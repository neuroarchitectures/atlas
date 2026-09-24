# ESMM

## Overview

Entire Space Multi-task Model: conversion-rate models trained only on clicked impressions suffer sample-selection bias and data sparsity. ESMM trains a pCTR and a pCVR tower (sharing embeddings) over every impression, supervising the product pCTCVR = pCTR x pCVR, so the CVR tower learns from the full space without ever needing click-only labels.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Entire Space Multi-task Model: conversion-rate models trained only on clicked impressions suffer sample-selection bias and data sparsity.

## Key Characteristics

- The two towers share one embedding table, which also eases the data-sparsity problem for the CVR tower.
- pCVR is never supervised directly: the losses are on pCTR and on the pCTCVR product, both defined over all impressions.
- A staple of e-commerce ads ranking; pairs with the multi-task rankers [mmoe](../mmoe/) and [ple](../ple/).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CVR / post-click conversion |
| Towers | Shared embeddings feed a pCTR and a pCVR tower |
| Objective | pCTCVR = pCTR x pCVR, supervised over all impressions |
| Key idea | Train CVR over the entire space to kill sample-selection bias |

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
