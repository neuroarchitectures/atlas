# Candidate-in-Sequence

## Design Philosophy

Target-aware interest weighting normally needs a separate attention unit (target-attention-din). BST's alternative: *append the candidate item to the behavior sequence* and let self-attention itself compute how each historical click relates to the target — the target attends to the history through the standard mechanism.

## Functionality

- Sequence = historical items + candidate as the final element; transformer encoder over the sequence; take the candidate position's output (or mean pool) as user interest → CTR MLP.

## Used By

| Model | Role |
|-------|------|
| BST (Behavior Sequence Transformer) | Item embed 64-d, seq len 50, 1 encoder layer (8 heads) → user tower |

## Features

- **Zero extra modules** — target-awareness is free with the encoder.
- **Symmetric attention** — candidate and history interact bidirectionally.

## Evolution

- **Predecessor**: target-attention-din (DIN activation unit).
- **Successor**: generative recommenders (HSTU) scale the same in-sequence formulation.
