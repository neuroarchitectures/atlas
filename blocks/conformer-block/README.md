# Conformer Block (Conv + Attention)

## Design Philosophy

Combine convolution (local patterns) with self-attention (global patterns) in a single block. The Conformer module interleaves conv and attention for speech processing.

## Functionality

1. Feed-Forward module (half-step). 2. Multi-head self-attention. 3. Convolution module (depthwise conv + pointwise conv). 4. Feed-Forward module (half-step). 5. LayerNorm. The half-step FFN uses 0.5x residual scaling.

## Used By

Conformer (speech recognition) | ASR systems | Audio classification

## Features

- **Local + global**: Conv captures local, attention captures global patterns.
- **Macaron structure**: Two FFN modules sandwiching attention and conv.
- **Proven**: State-of-the-art on speech recognition benchmarks.

## Evolution

Predecessor: Transformer, Convolution-augmented Transformer. Successor: Branchformer, E-Branchformer.
