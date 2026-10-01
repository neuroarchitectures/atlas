# Architecture: Contractive Autoencoder

## Motivation

Denoising autoencoders learn robustness to specific noise types. Contractive autoencoders learn robustness to all infinitesimal perturbations by penalizing the Frobenius norm of the encoder's Jacobian.

## Core Idea

Add a penalty term equal to the squared Frobenius norm of the Jacobian of the encoder mapping, forcing the learned representation to be locally flat — insensitive to small changes in the input.

## Architecture

### Overview

![contractive autoencoder architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [784] |
| 2 | Encoder | `linear` | outFeatures: 128, inFeatures: 784 |
| 3 | Sigmoid | `sigmoid` |  |
| 4 | Contracted Latent | `identity` | Jacobian penalty |
| 5 | Decoder | `linear` | outFeatures: 784, inFeatures: 128 |
| 6 | Reconstruction | `output` |  |

</details>

### Components

1. **Encoder** — Maps input to latent representation. 2. **Jacobian contraction penalty** — ||J_f(x)||²_F where J_f is the Jacobian of the encoder. This penalizes the sensitivity of the latent representation to input perturbations. 3. **Decoder** — Reconstructs the input from the contracted representation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Add a penalty term equal to the squared Frobenius norm of the Jacobian of the encoder mapping, forcing the learned representation to be locally flat — insensitive to small changes in the input.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Denoising Autoencoder. Related: Sparse Autoencoder. Successor: Variational Autoencoder.

## References

- Rifai et al. 2011
