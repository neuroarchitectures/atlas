# Straight-Through Gumbel-Softmax

## Design Philosophy

Discrete sampling (choose a category, include an edge, pick an action) has no gradient. The straight-through Gumbel-softmax trick fakes one: sample a hard one-hot in the forward pass, but backprop through the continuous Gumbel-softmax relaxation — with temperature τ controlling how discrete the relaxation is.

## Functionality

- `y_hard = onehot(argmax(log π + g))` in forward; gradient of `softmax((log π + g)/τ)` in backward (reparameterized Gumbel noise g).
- Temperature annealing τ → hardens toward true discrete sampling.

## Used By

| Model | Role |
|-------|------|
| CoGNN | Sampling each node's communication action per layer |
| MolGAN | One-shot discrete graph generation (dense X/A tensors) |
| NetGAN | Differentiable node sampling at each generated walk step |

## Features

- **End-to-end discrete training** — the standard bridge to categorical latent decisions.
- **Bias-variance knob** — τ trades gradient fidelity against sampling sharpness.

## Evolution

- **Predecessor**: REINFORCE (graphgan's generator update is the RL alternative), Gumbel-max trick.
- **Related**: codebook-quantization (wav2vec2) — the same trick for codebook selection; VQ-VAE straight-through estimators.
