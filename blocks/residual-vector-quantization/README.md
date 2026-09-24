# Residual Vector Quantization (RVQ)

## Design Philosophy

A cascade of codebooks where each quantizes the **residual** the previous one left. The philosophy: a single codebook can't capture fine detail; a cascade of coarse-to-fine codebooks reconstructs audio at very low bitrate. The discrete codes are what audio LLMs (MusicGen, etc.) actually generate — it is why "audio language model" is possible.

## Functionality

1. **Encode**: Input (e.g., conv encoder output) → continuous latent `z`.
2. **Codebook 1**: Quantize `z` to nearest codebook entry `c_1`; residual `r_1 = z - c_1`.
3. **Codebook 2**: Quantize `r_1` to `c_2`; residual `r_2 = r_1 - c_2`.
4. **...N codebooks**: Each quantizes the previous residual.
5. **Reconstruct**: `ẑ = c_1 + c_2 + ... + c_N`.
6. **Decode**: `ẑ` → mirrored decoder → audio.
- Straight-through estimator for gradient flow through the quantization.

## Used By

| Model | Role |
|-------|------|
| EnCodec (Meta) | The audio tokenizer behind MusicGen, audio LLMs |
| SoundStream | Google's audio codec |
| AudioLM, MusicGen | Generate the discrete codes autoregressively |

## Features

- **Low bitrate**: A few codebooks reconstruct audio at ~1–6 kbps.
- **Discrete tokens**: The codes are the "language" audio LLMs model.
- **Coarse-to-fine**: Each codebook adds detail the previous missed.

## Evolution

- **Predecessor**: VQ-VAE (single codebook for images); neural audio codecs.
- **Successor**: Higher-rate variants; language-model-over-audio-codes architectures (MusicGen, AudioLM).
