# Fourier Attention (FEDformer)

## Design Philosophy

Long time-series are dominated by a few low-frequency modes. Attending to every time step is wasteful when a sparse subset of Fourier components captures the global structure. FEDformer performs attention in the frequency domain, keeping only the top-k modes.

## Functionality

Project the sequence into the frequency domain with an FFT, perform attention on a randomly selected subset of the lowest-frequency modes (low-frequency + random-mode variants), then map back with the inverse FFT:

```
q, k, v = FFT(Q), FFT(K), FFT(V)
attention over selected modes -> multiply
out = IFFT( attention ) * v_freq
```

A frequency-domain MLP variant applies a linear projection in Fourier space instead of attention.

## Used By

FEDformer

## Features

- **Linear complexity**: attention cost depends on the number of kept modes, not sequence length.
- **Global receptive field**: low-frequency modes summarise the whole sequence.
- **Low-rank structure**: the random-mode variant provably approximates full attention.
- **Seasonal-trend friendly**: pairs with a decomposition block.

## Evolution

Predecessor: Autoformer (auto-correlation in the time domain). Sibling: Fourier positional encoding, FNet. Successor: frequency-domain token mixing in efficient long-sequence models.
