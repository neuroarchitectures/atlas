# DIEN

## Overview

Deep Interest Evolution Network: the sequel to DIN. After a target-aware attention scores the behaviour history, a GRU-based interest-evolution layer models how the user's interest moves over time before the CTR head. Captures temporal drift that DIN's single attention pool cannot.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Deep Interest Evolution Network: the sequel to DIN.

## Key Characteristics

- Builds directly on [din](../din/): same target-aware activation idea, with a recurrent interest-evolution layer added on top.
- Reference topology: behaviour attention into a GRU, concatenated with the candidate embedding before the MLP head.
- The full paper adds an auxiliary loss on the interest-extraction GRU and an attention-gated AUGRU; this graph shows the core evolution path.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CTR with user-behavior sequence |
| Interest attention | Target-aware scoring of the behaviour sequence |
| Interest evolution | GRU over the attended interests |
| Head | Concat evolved interest + candidate then MLP |
| Key idea | Model how interest drifts, not just which items matter |

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
