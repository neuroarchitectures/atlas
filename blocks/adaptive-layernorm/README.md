# Adaptive LayerNorm (adaLN / adaLN-Zero)

## Design Philosophy

Condition the LayerNorm's scale and shift on an external signal (timestep, class label). The philosophy: in conditional generation (diffusion), the conditioning signal should enter through *normalization*, not cross-attention — it is cheaper and touches every layer uniformly. adaLN-Zero further initializes the conditioning MLP to zero, so the block starts as identity (stable diffusion training).

## Functionality

- `cond = MLP(timestep_emb + class_emb)` → produces `(scale, shift, residual_gate)` per block.
- `adaLN(x) = scale ⊙ LayerNorm(x) + shift`.
- **adaLN-Zero**: The MLP's final weights are init to zero, so `scale=1, shift=0, gate=0` at init → the block is identity → the deep stack starts as a pure residual passthrough.

## Used By

| Model | Role |
|-------|------|
| DiT-XL/2 | adaLN-Zero conditioning on timestep + class |
| Stable Diffusion 3 | DiT backbone with adaLN |
| PixArt | adaLN conditioning |
| Sora (video) | DiT-style adaLN |

## Features

- **Cheap conditioning**: A small MLP produces (γ, β) — no cross-attention cost.
- **Uniform injection**: Conditioning touches every layer via norm, not just cross-attn layers.
- **Zero-init stability**: adaLN-Zero makes deep DiT stacks trainable from scratch.

## Evolution

- **Predecessor**: Conditional BatchNorm (StyleGAN); FiLM (feature-wise modulation).
- **Successor**: Used broadly in DiT-class diffusion models; cross-attention remains for text, adaLN for timestep.
