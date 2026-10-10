# Classifier-Free Guidance

## Design Philosophy

Instead of training a separate classifier for guided diffusion, train the model with and without conditioning (randomly drop conditioning). At inference, extrapolate from conditional to unconditional.

## Functionality

eps_guided = eps_uncond + w * (eps_cond - eps_uncond), where w >= 1 is the guidance scale. During training, randomly drop the condition (10-20% of the time) to learn both conditional and unconditional.

## Used By

DALL-E 2 | Stable Diffusion | Imagen | Most modern diffusion models

## Features

- **No classifier needed**: Simpler than classifier guidance.
- **Controllable**: Guidance scale w controls fidelity vs. diversity.
- **Single model**: One model for conditional and unconditional.

## Evolution

Predecessor: Classifier guidance. Successor: Guidance distillation, classifier-free + classifier guidance.
