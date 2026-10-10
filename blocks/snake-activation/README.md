# Snake Activation (Periodic)

## Design Philosophy

Standard activations (ReLU, GELU, SiLU) are monotonic and non-periodic, but audio signals and their harmonics are intrinsically periodic. A periodic, learnable activation gives the generator a built-in inductive bias for periodic structure, letting it synthesize waveforms far more faithfully than pointwise monotonic nonlinearities.

## Functionality

A per-channel periodic activation with two learnable scalars:

```
snake(x) = x + (1 / a) * sin^2(a * x)
```

where `a` controls the frequency. The identity term keeps large-magnitude responses (non-saturating), while the `sin^2` term injects periodicity. A periodic variant, `x + (1/a) * sin^2(a*(x + 1/a))`, adds a phase shift.

## Used By

BigVGAN

## Features

- **Periodic inductive bias**: matches the spectral nature of audio.
- **Learnable frequency**: `a` is trained per channel, adapting to each band.
- **Non-saturating**: the identity term avoids vanishing gradients for large inputs.
- **Generates harmonics**: the periodic term lets a single unit produce overtones, key for realistic audio.

## Evolution

Predecessor: SiLU/Swish (non-periodic, no frequency prior). Sibling: anti-aliased activation, which filters the nonlinearity's high-frequency artifacts. Successor: anti-aliased periodic activations used in modern neural vocoders.
