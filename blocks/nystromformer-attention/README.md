# Nyströmformer Attention

## Design Philosophy

Approximate the full attention matrix using the Nyström method: sample m landmark points, compute attention between Q/K and landmarks, and reconstruct the full attention matrix.

## Functionality

Select m landmark queries/keys (via uniform sampling or DPP). Compute Q_landmark K^T and K_landmark V, then approximate full attention as Q K_landmark^T (K_landmark K^T)^{-1} K^T V.

## Used By

Nyströmformer | Efficient long-context models

## Features

- **O(L m)**: Linear in L, where m is the number of landmarks.
- **Nyström approximation**: Well-studied matrix approximation technique.
- **No hashing**: Simpler than LSH-based methods.

## Evolution

Predecessor: Reformer (LSH). Successor: Efficient attention variants.
