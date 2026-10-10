# Architecture: ESM-2

## Motivation

Protein sequences are like natural language — evolutionary constraints create patterns. ESM-2 applies masked language modeling to 250M protein sequences, learning representations that capture structure and function without labels.

## Core Idea

A transformer encoder (up to 15B params) trained with masked language modeling on 250M protein sequences. The learned embeddings capture evolutionary information, secondary structure, and contact maps. Enables zero-shot prediction of mutation effects.

## Architecture

### Overview

![esm-2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Amino Acid Sequence | `input` | up to 1024 residues |
| 2 | Token Embedding | `linear` | 20 amino acids -> 1280 dim |
| 3 | Transformer Layers | `linear` | up to 48 layers |
| 4 | LayerNorm | `identity` | final norm |
| 5 | LM Head | `linear` | 1280 -> 33 vocab |
| 6 | Residue Predictions | `output` | masked residue prediction |

</details>

### Components

1. **Masked language modeling** — 15% of residues masked, predict from context. 2. **250M sequences** — trained on UniRef database. 3. **Scale variants** — 8M, 35M, 150M, 650M, 3B, 15B params. 4. **Rotary embeddings** — RoPE positional encoding. 5. **Emergent structure** — attention patterns reflect 3D contacts. 6. **Zero-shot** — predicts mutation effects without fine-tuning.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A transformer encoder (up to 15B params) trained with masked language modeling on 250M protein sequences. The learned em...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ESM-1b, MSA Transformer. Successor: ESMFold (structure), ESM3.

## References

- Lin et al. 2023
