# Reformer Attention (LSH)

## Design Philosophy

Use Locality-Sensitive Hashing (LSH) to group similar queries and keys into buckets. Only compute attention within buckets, reducing from O(L^2) to O(L log L).

## Functionality

Hash Q and K using random projections. Tokens in the same bucket attend to each other. Use multiple hash functions for better coverage. Sort by hash, then process in chunks.

## Used By

Reformer | Long-document processing | Memory-efficient transformers

## Features

- **O(L log L)**: Near-linear complexity.
- **Hash-based**: No learned parameters for the sparse pattern.
- **Multiple hashes**: Reduces collision-induced information loss.

## Evolution

Predecessor: Sparse attention. Successor: Performer, Nyströmformer.
