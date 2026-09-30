# Architecture: H3

## Motivation

State space models (S4) show promise for long-sequence modeling but lack the inductive biases needed for language. H3 identifies that the key missing ingredient is the ability to perform content-based selection — recalling specific tokens from the past based on current context.

## Core Idea

H3 combines two SSMs: a shift-SSM (for local context/shift) and a diagonal SSM (for long-range memory), connected via multiplicative gating. This architecture mirrors the structure of attention (query-key-value) but replaces the quadratic attention matrix with efficient SSM computation.

## Architecture

### Overview

![h3 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (18 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 1024] |
| 2 | Embedding | `embedding` | numEmbeddings: 50257, embeddingDim: 1024 |
| 3 | q_proj | `linear` | outFeatures: 1024, inFeatures: 1024 |
| 4 | k_proj | `linear` | outFeatures: 1024, inFeatures: 1024 |
| 5 | v_proj | `linear` | outFeatures: 1024, inFeatures: 1024 |
| 6 | Shift-SSM | `custom` | type: shift_ssm |
| 7 | Shift-SSM | `custom` | type: shift_ssm |
| 8 | Diagonal SSM | `custom` | type: diagonal_ssm, stateSize: 64 |
| 9 | q ⊙ k | `multiply` |  |
| 10 | Diagonal SSM | `custom` | type: diagonal_ssm, stateSize: 64 |
| 11 | output ⊙ v | `multiply` |  |
| 12 | out_proj | `linear` | outFeatures: 1024, inFeatures: 1024 |
| 13 | Residual | `add` |  |
| 14 | norm_ffn | `rmsNorm` | normalizedShape: 1024 |
| 15 | FFN | `swiglu` | embedDim: 1024, intermediateSize: 2816 |
| 16 | Residual | `add` |  |
| 17 | lm_head | `linear` | outFeatures: 50257, inFeatures: 1024 |
| 18 | Output | `output` |  |

</details>

H3 combines two SSMs: a shift-SSM (for local context/shift) and a diagonal SSM (for long-range memory), connected via multiplicative gating. This architecture mirrors the structure of attention (query-key-value) but replaces the quadratic attention matrix with efficient SSM computation.

### Components

2. **Embedding** (`embedding`, scope: `embeddings`) — Params: numEmbeddings: 50257, embeddingDim: 1024
3. **q_proj** (`linear`, scope: `layer.0.h3`) — Params: outFeatures: 1024, inFeatures: 1024
4. **k_proj** (`linear`, scope: `layer.0.h3`) — Params: outFeatures: 1024, inFeatures: 1024
5. **v_proj** (`linear`, scope: `layer.0.h3`) — Params: outFeatures: 1024, inFeatures: 1024
6. **Shift-SSM** (`custom`, scope: `layer.0.h3`) — Params: type: shift_ssm
7. **Shift-SSM** (`custom`, scope: `layer.0.h3`) — Params: type: shift_ssm
8. **Diagonal SSM** (`custom`, scope: `layer.0.h3`) — Params: type: diagonal_ssm, stateSize: 64
9. **q ⊙ k** (`multiply`, scope: `layer.0.h3`) — Params: none
10. **Diagonal SSM** (`custom`, scope: `layer.0.h3`) — Params: type: diagonal_ssm, stateSize: 64
11. **output ⊙ v** (`multiply`, scope: `layer.0.h3`) — Params: none
12. **out_proj** (`linear`, scope: `layer.0.h3`) — Params: outFeatures: 1024, inFeatures: 1024
13. **Residual** (`add`, scope: `layer.0`) — Params: none
14. **norm_ffn** (`rmsNorm`, scope: `layer.0.ffn`) — Params: normalizedShape: 1024
15. **FFN** (`swiglu`, scope: `layer.0.ffn`) — Params: embedDim: 1024, intermediateSize: 2816
16. **Residual** (`add`, scope: `layer.0`) — Params: none
17. **lm_head** (`linear`, scope: `model`) — Params: outFeatures: 50257, inFeatures: 1024

### Data Flow

The architecture processes input through a sequence of 18 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to H3.

## Evolution

H3 builds on S4 (structured state spaces) and is a direct predecessor of Hyena and Mamba. It demonstrates that SSM-based architectures can match Transformers on language tasks by introducing the query-key-value structure into SSMs. Mamba later simplified this into a single selective SSM block.

## Source

- **Paper:** arXiv:2212.14052
- **Year:** 2022
- **Authors:** Fu et al.
