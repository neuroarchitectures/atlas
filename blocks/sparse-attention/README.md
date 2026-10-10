# Sparse Attention (Sparse Transformer)

## Design Philosophy

Full attention is O(L^2). Sparse attention restricts each token to attend to a subset of positions (strided + fixed), reducing to O(L sqrt(L)) while maintaining long-range connectivity.

## Functionality

Two patterns: (1) strided attention (attend to every k-th position), (2) fixed attention (attend to specific positions). Combined, they cover both local and long-range dependencies.

## Used By

Sparse Transformer | Longformer (window + global) | BigBird (random + window + global)

## Features

- **O(L sqrt(L))**: Subquadratic complexity.
- **Flexible patterns**: Can be strided, local, global, or random.
- **Universal approximation**: With appropriate patterns, approximates full attention.

## Evolution

Predecessor: Full attention. Successor: Longformer, BigBird, FlashAttention.
