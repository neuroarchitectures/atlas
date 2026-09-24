# Neural Synchronization Representation

## Design Philosophy

A latent *state* discards the history of how neurons co-activated. CTM's representation is the inner-product matrix of post-activation histories — `S_t = Z_t Z_tᵀ` — the neurons' synchronization structure, with learnable per-pair exponential decay r_ij and random pair sub-sampling to keep D² tractable.

## Functionality

- Collect post-activation histories across T internal ticks; S_t = Z_t Z_tᵀ with decay r_ij; subsample to D_out output pairs and D_action attention-query pairs.
- Loss read at t₁ = argmin(L) and t₂ = argmax(certainty) across ticks; queries used for cross-attention against frozen FeatureExtractor K/Vs.

## Used By

| Model | Role |
|-------|------|
| Continuous Thought Machines | Representation + attention queries for perception and decision |

## Features

- **History as representation** — timing of computation becomes part of the output.
- **Adaptive readout timing** — the model chooses *when* it has finished thinking.

## Evolution

- **Predecessor**: static final states; synchrony-based codes in neuroscience.
- **Companion**: neuron-level-model — the histories come from NLMs.
