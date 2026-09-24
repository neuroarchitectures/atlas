# Sinusoidal Positional Encoding

## Design Philosophy

Positions should be injected without learned parameters and should generalize to unseen lengths. Fixed sin/cos waves at geometrically spaced wavelengths let the model represent relative positions as linear functions of absolute encodings — the original Transformer's solution.

## Functionality

- `PE(pos, 2i) = sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(...)`; added to token embeddings of both streams.
- Wavelengths geometric from 2π to 10000·2π — each dimension a different frequency.

## Used By

| Model | Role |
|-------|------|
| Transformer (2017) | d_model=512, added to encoder and decoder embeddings |

## Features

- **Parameter-free, length-extensible**.
- **Relative-position linearity** — PE(pos+k) is a linear transform of PE(pos).

## Evolution

- **Predecessor**: learned positional encoding (BERT/GPT) — traded extensibility for flexibility.
- **Successor**: relative-position-bias (T5), rope (rotary — relative by construction), ALiBi.
