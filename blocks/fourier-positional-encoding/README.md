# Fourier Positional Encoding (Coordinate Mapping)

## Design Philosophy

An MLP is spectrally biased toward low frequencies — feed it raw 3D coordinates and it renders blurry scenes. Map coordinates through fixed multi-frequency sinusoids (Fourier features) so the MLP can represent high-frequency spatial detail.

## Functionality

- `γ(p) = [sin(2^k π p), cos(2^k π p)]` for k = 0..L−1 per coordinate (NeRF: L=10 for location, 4 for view direction).
- NeRF's 5D input (x,y,z,θ,φ) → high-dim embedding → density + view-dependent color MLP.

## Used By

| Model | Role |
|-------|------|
| NeRF | High-frequency scene detail from posed images |

## Features

- **Spectral bias fix** — the enabling trick of neural fields.
- **Frequency count = detail knob** (L too high → aliasing artifacts).

## Evolution

- **Predecessor**: Tancik et al. Fourier features (2020); SIREN (sinusoidal activations) as the alternative.
- **Successor**: hash-grid encodings (instant-NGP) replace global Fourier features with learned sparse tables.
