# Architecture: Hyena

## Motivation

Transformers achieve strong performance but scale quadratically with sequence length. Prior subquadratic alternatives (linear attention, gated convolutions, SSMs) underperform on language tasks. Hyena aims to match Transformer quality at long context while being subquadratic.

## Core Idea

Replace self-attention with a hierarchy of implicitly parameterized long convolutions interleaved with multiplicative gating. The key insight is that attention can be recast as structured matrix multiplication, and long convolutions parameterized by neural networks can approximate this structure at subquadratic cost.

## Architecture

### Overview

![hyena architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (14 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 1024] |
| 2 | Token Embedding | `embedding` | numEmbeddings: 50257, embeddingDim: 1024 |
| 3 | in_proj (H) | `linear` | outFeatures: 2048, inFeatures: 1024 |
| 4 | in_proj (x) | `linear` | outFeatures: 2048, inFeatures: 1024 |
| 5 | Hyena Filter (H) | `custom` | type: long_conv, kernelLen: 1024 |
| 6 | Gating Branch (x) | `custom` | type: gating |
| 7 | Hadamard Product | `multiply` |  |
| 8 | out_proj | `linear` | outFeatures: 1024, inFeatures: 2048 |
| 9 | Residual | `add` |  |
| 10 | norm_ffn | `rmsNorm` | normalizedShape: 1024 |
| 11 | FFN | `swiglu` | embedDim: 1024, intermediateSize: 2816 |
| 12 | Residual | `add` |  |
| 13 | lm_head | `linear` | outFeatures: 50257, inFeatures: 1024 |
| 14 | Output | `output` |  |

</details>

Replace self-attention with a hierarchy of implicitly parameterized long convolutions interleaved with multiplicative gating. The key insight is that attention can be recast as structured matrix multiplication, and long convolutions parameterized by neural networks can approximate this structure at subquadratic cost.

### Components

2. **Token Embedding** (`embedding`, scope: `embeddings`) — Params: numEmbeddings: 50257, embeddingDim: 1024
3. **in_proj (H)** (`linear`, scope: `layer.0.hyena`) — Params: outFeatures: 2048, inFeatures: 1024
4. **in_proj (x)** (`linear`, scope: `layer.0.hyena`) — Params: outFeatures: 2048, inFeatures: 1024
5. **Hyena Filter (H)** (`custom`, scope: `layer.0.hyena`) — Params: type: long_conv, kernelLen: 1024
6. **Gating Branch (x)** (`custom`, scope: `layer.0.hyena`) — Params: type: gating
7. **Hadamard Product** (`multiply`, scope: `layer.0.hyena`) — Params: none
8. **out_proj** (`linear`, scope: `layer.0.hyena`) — Params: outFeatures: 1024, inFeatures: 2048
9. **Residual** (`add`, scope: `layer.0`) — Params: none
10. **norm_ffn** (`rmsNorm`, scope: `layer.0.ffn`) — Params: normalizedShape: 1024
11. **FFN** (`swiglu`, scope: `layer.0.ffn`) — Params: embedDim: 1024, intermediateSize: 2816
12. **Residual** (`add`, scope: `layer.0`) — Params: none
13. **lm_head** (`linear`, scope: `model`) — Params: outFeatures: 50257, inFeatures: 1024

### Data Flow

The architecture processes input through a sequence of 14 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Hyena.

## Evolution

Hyena builds on structured state space models (S4, H3) and replaces their recurrence with implicit long convolutions. It demonstrates that attention-free architectures can match Transformer quality on language tasks while being subquadratic. Successors include HyenaDNA (genomic), StripedHyena, and Mamba.

## Source

- **Paper:** arXiv:2302.10866
- **Year:** 2023
- **Authors:** Nguyen et al.
