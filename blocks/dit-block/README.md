# DiT Block (Diffusion Transformer)

## Design Philosophy

Replace the diffusion U-Net with a **plain Transformer over latent patches**, conditioned by **adaptive LayerNorm (adaLN-Zero)**. The philosophy: diffusion should scale like LLMs do, and Transformers scale better than U-Nets. adaLN-Zero (timestep + class conditioning through normalization, zero-init) makes deep DiT stacks trainable from scratch. DiT is the backbone of Stable Diffusion 3, PixArt, and Sora-class video models.

## Functionality

- **Patchify**: Latent (e.g., 4×32×32) → 2×2 patches → N tokens.
- **N DiT blocks**, each:
  1. `cond = MLP(timestep_emb + class_emb)` → `(scale, shift, gate_attn, gate_ffn)`.
  2. `adaLN(x) = scale ⊙ LayerNorm(x) + shift`.
  3. Multi-head self-attention → `gate_attn ⊙ Attn(adaLN(x)) + x`.
  4. SwiGLU FFN → `gate_ffn ⊙ FFN(adaLN(x)) + x`.
- **adaLN-Zero**: The conditioning MLP's final weights init to zero → gates start at 0 → block is identity.
- **Output**: LayerNorm + linear → per-patch noise prediction.

## Used By

| Model | Role |
|-------|------|
| DiT-XL/2 | 28 blocks, 1152-dim, 675M params |
| Stable Diffusion 3 | DiT backbone |
| PixArt | DiT with adaLN |
| Sora (video) | DiT-class backbone |

## Features

- **Transformer scalability**: Diffusion scales like LLMs — more data, more compute, better quality.
- **adaLN-Zero conditioning**: Timestep + class enter through norm, zero-init for stability.
- **Latent patches**: Same patchify as ViT, applied to the VAE latent.

## Evolution

- **Predecessor**: Diffusion U-Net (SD 1.x/2.x); ViT (patch embedding).
- **Successor**: Scalable DiT variants; video DiT (Sora); the dominant diffusion backbone going forward.
