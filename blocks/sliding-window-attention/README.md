# Sliding-Window Attention

## Design Philosophy

Each token attends only to a fixed **window** of `w` nearby tokens instead of the whole sequence. Cost drops from O(n²) to O(n·w). The philosophy: most attention mass is local, and stacking L layers grows the effective receptive field to ~L·w, so a 4K window over 32 layers reaches far longer context than the window alone suggests.

## Functionality

- **Local attention**: `localAttention` with `windowSize = w`. Each query attends to `w` neighbors on each side (or `w` total).
- Same Q/K/V projections and softmax as full attention; only the mask changes.
- **Receptive field growth**: Layer `l` can see `l·w` tokens back; stacking recovers long-range reach.

## Used By

| Model | Role |
|-------|------|
| Longformer | Windowed attention + a few global tokens |
| Mistral-7B | 4K sliding window (the basis of its long context) |
| Mixtral | Same sliding window + sparse MoE |
| Big Bird | Window + random + global (theoretical Turing-complete) |

## Features

- **Linear in sequence length**: O(n·w), not O(n²).
- **Drop-in**: Same I/O shape as full attention — swap one node.
- **Receptive field via depth**: Stacking windows grows reach without growing per-layer cost.

## Evolution

- **Predecessor**: Full (dense) attention.
- **Successor**: Sparse attention (block-sparse, top-k selection); hybrid window + global (Big Bird).
