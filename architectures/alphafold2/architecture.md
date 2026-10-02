# Architecture: AlphaFold 2

## Motivation

Protein structure prediction from sequence is a 50-year-old grand challenge. AlphaFold 2 achieves near-experimental accuracy by combining evolutionary constraints, attention, and geometric structure modules.

## Core Idea

Input sequence is aligned to create MSA and pair representations. The Evoformer (48 layers of axial attention) jointly processes MSA and pair features. A structure module uses invariant point attention to predict 3D coordinates with SE(3) equivariance.

## Architecture

### Overview

![alphafold2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Amino Acid Sequence | `input` | up to 1024 residues |
| 2 | MSA + Pair Feature | `identity` | MSA + pair embeddings |
| 3 | Evoformer | `linear` | 48 blocks, axial attention |
| 4 | Structure Module | `linear` | IPA + rigid-body frames |
| 5 | 3D Coordinates | `output` | per-residue coordinates |

</details>

### Components

1. **MSA representation** — multiple sequence alignment captures evolutionary co-variation. 2. **Pair representation** — pairwise residue features (distance/orientation). 3. **Evoformer** — 48 blocks of alternating row and column attention on MSA + pair. 4. **Invariant Point Attention (IPA)** — SE(3)-equivariant attention in 3D. 5. **Structure module** — predicts rigid-body frames per residue. 6. **Recycling** — iterative refinement over previous predictions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Input sequence is aligned to create MSA and pair representations. The Evoformer (48 layers of axial attention) jointly p...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AlphaFold (2018, distance prediction). Successor: AlphaFold-Multimer, ESMFold (single-sequence).

## References

- Jumper et al. 2021 (Nature)
