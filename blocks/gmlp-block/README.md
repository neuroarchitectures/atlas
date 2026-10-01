# gMLP Block (Spatial Gating)

## Design Philosophy

MLP-Mixer uses fixed token-mixing. gMLP replaces it with a learnable spatial gating unit (SGU) that can model input-dependent spatial interactions.

## Functionality

1. LayerNorm, channel MLP (expand). 2. Split into two halves. 3. Spatial gating: Z = X1 * SGD(X2), where SGD is a linear layer over the spatial dimension. 4. Channel MLP (project), residual.

## Used By

gMLP | MLP-based sequence models

## Features

- **Input-dependent gating**: The SGU adapts to the input.
- **No attention**: Still a pure MLP architecture.
- **Competitive**: Matches Transformer on some tasks but underperforms on NLP.

## Evolution

Predecessor: MLP-Mixer. Successor: ResMLP, CycleMLP.
