# Neuron-Level Model (NLM)

## Design Philosophy

Every neuron shares one activation function — a constraint biology doesn't have. The Continuous Thought Machine gives *each latent neuron its own tiny depth-1 MLP* over its private M-length pre-activation history, replacing the shared activation with a learned, temporally-aware per-neuron dynamics.

## Functionality

- Per neuron d: NLM_d(a_{t-M:t}^d) → post-activation; M ≈ 10–100 ticks of history.
- Pre-activations come from a shared synapse MLP: `a_t = f(concat(z_t, o_t))`.

## Used By

| Model | Role |
|-------|------|
| Continuous Thought Machines | One NLM per latent neuron of the D-dim state; inner ticks give the model internal computation time |

## Features

- **Per-neuron temporal processing** — history depth is a per-neuron computational resource.
- **Internal ticks** — the model can "think longer" on harder inputs.

## Evolution

- **Predecessor**: shared activations; Liquid Time-Constant networks.
- **Companion**: neural-synchronization — the readout built on these histories.
