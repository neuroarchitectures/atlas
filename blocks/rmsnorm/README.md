# RMSNorm

## Design Philosophy

LayerNorm without the mean subtraction. The philosophy: the mean-centering step in LayerNorm is unnecessary for modern LLMs and adds a costly reduction across the hidden dim. RMSNorm only re-scales by the root-mean-square, which is cheaper and empirically equivalent in quality. It is the norm of choice for every modern decoder LLM.

## Functionality

`RMSNorm(x) = x / RMS(x) · γ`, where `RMS(x) = √(mean(x²) + ε)` and `γ` is a learnable per-channel scale.

- No mean subtraction (saves one pass over the hidden dim).
- No bias (just a scale `γ`).
- Pre-norm placement: `x + Sublayer(RMSNorm(x))`.

## Used By

| Model | Role |
|-------|------|
| LLaMA-2/3 | Pre-norm before attention and FFN |
| Mistral / Mixtral | Same |
| Qwen, Yi, Phi, Gemma | Same |
| DeepSeek-V2/V3 | Same |

## Features

- **Cheaper than LayerNorm**: ~7–64× faster on the norm op (depending on dim).
- **Same quality**: Empirically interchangeable with LayerNorm for LLM training.
- **Pre-norm**: Enables deep stacks with stable gradients.

## Evolution

- **Predecessor**: LayerNorm (Ba et al., 2016); BatchNorm (Ioffe & Szegedy, 2015).
- **Successor**: The modern default; variants like DiT's adaptive LayerNorm (adaLN) add conditioning.
