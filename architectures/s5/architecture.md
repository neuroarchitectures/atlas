# Architecture: S5

## Motivation

S4 achieves strong results on long-sequence benchmarks but uses complex duality-based initialization and multiple independent SISO (single-input single-output) SSMs, making it computationally and conceptually complex. S5 simplifies this.

## Core Idea

Replace S4's multiple independent SISO SSMs with a single multi-input multi-output (MIMO) SSM. Use a parallel scan algorithm for efficient computation. This simplification reduces parameter count and computation while maintaining the expressiveness needed for long-range modeling.

## Architecture

### Overview

![s5 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (14 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 1024] |
| 2 | Embedding | `embedding` | numEmbeddings: 50257, embeddingDim: 1024 |
| 3 | norm | `rmsNorm` | normalizedShape: 1024 |
| 4 | in_proj | `linear` | outFeatures: 2048, inFeatures: 1024 |
| 5 | MIMO SSM | `custom` | type: mimo_ssm, stateSize: 64, inputDim: 2048, outputDim: 2048 |
| 6 | Gating | `swish` |  |
| 7 | Gate | `multiply` |  |
| 8 | out_proj | `linear` | outFeatures: 1024, inFeatures: 2048 |
| 9 | Residual | `add` |  |
| 10 | norm_ffn | `rmsNorm` | normalizedShape: 1024 |
| 11 | FFN | `swiglu` | embedDim: 1024, intermediateSize: 2816 |
| 12 | Residual | `add` |  |
| 13 | lm_head | `linear` | outFeatures: 50257, inFeatures: 1024 |
| 14 | Output | `output` |  |

</details>

Replace S4's multiple independent SISO SSMs with a single multi-input multi-output (MIMO) SSM. Use a parallel scan algorithm for efficient computation. This simplification reduces parameter count and computation while maintaining the expressiveness needed for long-range modeling.

### Components

2. **Embedding** (`embedding`, scope: `embeddings`) — Params: numEmbeddings: 50257, embeddingDim: 1024
3. **norm** (`rmsNorm`, scope: `layer.0.ssm`) — Params: normalizedShape: 1024
4. **in_proj** (`linear`, scope: `layer.0.ssm`) — Params: outFeatures: 2048, inFeatures: 1024
5. **MIMO SSM** (`custom`, scope: `layer.0.ssm`) — Params: type: mimo_ssm, stateSize: 64, inputDim: 2048, outputDim: 2048
6. **Gating** (`swish`, scope: `layer.0.ssm`) — Params: none
7. **Gate** (`multiply`, scope: `layer.0.ssm`) — Params: none
8. **out_proj** (`linear`, scope: `layer.0.ssm`) — Params: outFeatures: 1024, inFeatures: 2048
9. **Residual** (`add`, scope: `layer.0`) — Params: none
10. **norm_ffn** (`rmsNorm`, scope: `layer.0.ffn`) — Params: normalizedShape: 1024
11. **FFN** (`swiglu`, scope: `layer.0.ffn`) — Params: embedDim: 1024, intermediateSize: 2816
12. **Residual** (`add`, scope: `layer.0`) — Params: none
13. **lm_head** (`linear`, scope: `model`) — Params: outFeatures: 50257, inFeatures: 1024

### Data Flow

The architecture processes input through a sequence of 14 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to S5.

## Evolution

S5 simplifies S4 by replacing multiple SISO SSMs with a single MIMO SSM and using a parallel scan. It is a predecessor to Mamba, which further simplified the architecture by making SSM parameters input-dependent.

## Source

- **Paper:** arXiv:2208.04933
- **Year:** 2022
- **Authors:** Smith et al.
