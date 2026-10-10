# Architecture: Swin V3

## Motivation

Swin Transformer V2 scaled to larger models but relative position bias and attention patterns needed improvement.

## Core Idea

Enhance Swin with improved relative position bias (log-spaced continuous bias), neighborhood attention, and better scaling strategies for large models.

## Architecture

### Overview

![swin-v3 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Patch Embedding | `patchify` | patchSize: 4 |
| 3 | Swin Stage 1 | `swin-block` | depth: 2 |
| 4 | Patch Merge | `patch-merge` |  |
| 5 | Swin Stage 2 | `swin-block` | depth: 2 |
| 6 | Patch Merge | `patch-merge` |  |
| 7 | Swin Stage 3 | `swin-block` | depth: 6 |
| 8 | Patch Merge | `patch-merge` |  |
| 9 | Swin Stage 4 | `swin-block` | depth: 2 |
| 10 | Global Avg Pool | `pooling` |  |
| 11 | Classifier | `linear` | outFeatures: 1000 |
| 12 | Class Logits | `output` |  |

</details>

### Components

1. **Patch embedding** — 4x4 patches to 96-dim. 2. **Swin stages** — Window-based multi-head self-attention with shifted windows. 3. **Patch merge** — 2x spatial downsample between stages. 4. **Improved relative position bias** — Log-spaced continuous interpolation. 5. **Neighborhood attention** — Dilated local attention for broader context.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Enhance Swin with improved relative position bias (log-spaced continuous bias), neighborhood attention, and better scali
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Swin-V2. Successor: (current).

## References

Liu et al. 2023
