# Cross-Dimension Attention (Crossformer)

## Design Philosophy

Multivariate time series vary along two axes: time and channels (variables). Standard attention attends across time only, ignoring cross-channel structure. Crossformer alternates attention over each axis — cross-time and cross-dimension — so both temporal and inter-variable dependencies are modelled.

## Functionality

After segmenting the series into patches and embedding them (two-stage), each layer applies:

- **Cross-Time attention**: self-attention across the time (patch) dimension for each variable.
- **Cross-Dimension attention**: self-attention across the variable dimension, with a router that reduces the number of channel queries for efficiency.

Alternating the two yields a "two-stage" representation, and the layer's complexity scales linearly with the number of variables.

## Used By

Crossformer

## Features

- **Two-axis modelling**: time and channel attention are explicit and separate.
- **Router**: bounds the cost of cross-dimension attention.
- **Two-stage embedding**: captures patch and group-level structure.
- **Hierarchical**: multiple scales of temporal resolution.

## Evolution

Predecessor: temporal-only attention (Informer, Autoformer). Sibling: channel-independent PatchTST. Successor: multivariate efficient transformers with explicit cross-channel mixing.
