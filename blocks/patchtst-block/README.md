# PatchTST Block (Channel-Independent Patch Transformer)

## Design Philosophy

Slice each time-series variable into **patches** (like ViT, but in time) and encode them **channel-independently** by a shared Transformer. The philosophy: patching cuts sequence length quadratically for attention and gives each token local semantic content; channel independence (one shared encoder per variable) beat channel-mixing on long-horizon benchmarks. The Transformer that made long-horizon time-series forecasting work.

## Functionality

- **Patch embed**: Each variable's 1D series → patches of size `P`, stride `S` → linear to `D`-dim tokens.
- **Channel independence**: The same Transformer encoder is applied to each variable's patches separately (no cross-variable mixing in the encoder).
- **Transformer blocks**: Pre-norm MHA + FFN, standard.
- **Head**: Flatten + linear → forecast horizon.

## Used By

| Model | Role |
|-------|------|
| PatchTST | The defining model for long-horizon forecasting |

## Features

- **Patching**: Quadratic sequence-length reduction for attention; local semantic tokens.
- **Channel independence**: Shared encoder per variable — better than channel-mixing.
- **Long-horizon**: Made multi-step forecasting competitive.

## Evolution

- **Predecessor**: Informer / Autoformer (Transformer-based, full-length attention); DLinear (simple linear).
- **Successor**: iTransformer (inverted — attend across variables, not time); TimeMixer (multi-scale MLP).
