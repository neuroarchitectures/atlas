# Architecture: ESMFold

## Motivation

AlphaFold 2 requires MSA construction which is slow (~minutes). ESMFold uses a pretrained protein language model (ESM-2) to predict structure from a single sequence in seconds.

## Core Idea

A pretrained protein language model (ESM-2, 15B parameters) extracts rich residue embeddings from a single sequence. A folding trunk (attention layers) processes these into pair features. A structure module predicts 3D coordinates.

## Architecture

### Overview

![esmfold architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Amino Acid Sequence | `input` | up to 1024 residues |
| 2 | Protein LM | `linear` | ESM-2, 15B params |
| 3 | Pair Feature | `linear` | residue embeddings -> pair |
| 4 | Folding Trunk | `linear` | attention layers |
| 5 | Structure Module | `linear` | IPA + frames |
| 6 | 3D Coordinates | `output` | per-residue coords |

</details>

### Components

1. **ESM-2 language model** — pretrained on 250M sequences, captures evolutionary information without MSA. 2. **Folding trunk** — 48 attention layers process single-sequence embeddings. 3. **Structure module** — invariant point attention for 3D coordinate prediction. 4. **No MSA needed** — 60x faster than AlphaFold 2. 5. **Slightly lower accuracy** — trades accuracy for speed.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A pretrained protein language model (ESM-2, 15B parameters) extracts rich residue embeddings from a single sequence. A f...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AlphaFold 2 (MSA-based), ESM-2 (language model). Successor: ESM3 (generative).

## References

- Lin et al. 2023
