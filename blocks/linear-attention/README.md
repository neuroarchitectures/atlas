# Linear Attention

## Design Philosophy

Replace the softmax kernel in attention with a linear kernel (e.g., phi(x) = elu(x) + 1). This allows computing (Q phi) (K phi)^T V, which is O(L d^2) instead of O(L^2 d).

## Functionality

phi(x) = elu(x) + 1 (or other non-negative feature). Attention = (phi(Q) phi(K)^T) V = phi(Q) (phi(K)^T V). The associative trick reduces complexity from O(L^2) to O(L).

## Used By

Linear Transformer | Performer (with random features) | Efficient long-context models

## Features

- **O(L) complexity**: Linear in sequence length.
- **Causal masking**: Possible via left-to-right cumulative sum.
- **Approximate**: Not exact softmax; quality depends on kernel choice.

## Evolution

Predecessor: Standard attention. Successor: FAVOR+ (Performer), FlashAttention (exact, efficient).
