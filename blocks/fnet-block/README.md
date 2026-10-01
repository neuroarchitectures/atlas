# FNet Block (Fourier Transform)

## Design Philosophy

Replace self-attention with a 2D Fourier Transform. The FFT mixes tokens (along the sequence dimension) and channels (along the feature dimension) with O(L log L) complexity and no parameters.

## Functionality

1. LayerNorm. 2. 2D FFT along (sequence, feature) dimensions. 3. Take real part. 4. LayerNorm, FFN, residual. The FFT replaces both self-attention and the token-mixing MLP.

## Used By

FNet | Efficient text models

## Features

- **No parameters**: The FFT is parameter-free.
- **O(L log L)**: Very efficient.
- **Surprisingly effective**: Matches BERT on some NLP tasks.

## Evolution

Predecessor: Transformer, MLP-Mixer. Successor: Fourier-based neural operators.
