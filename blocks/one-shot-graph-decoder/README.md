# One-Shot Probabilistic Graph Decoder

## Design Philosophy

Autoregressive graph generation is slow and order-dependent. Emit the *whole graph at once*: an MLP over the latent code produces a fully-connected probabilistic graph — edge-existence, edge-type, and node-type heads up to a maximum k nodes — and align the reconstruction to the target via approximate max-pooling graph matching inside the ELBO.

## Functionality

- z → MLP → k×k edge probabilities (+ edge/node type channels); Bernoulli reconstruction.
- Approximate max-pooling matching (~75 iterations) computes the node assignment matrix for loss alignment.

## Used By

| Model | Role |
|-------|------|
| GraphVAE | VAE with ECC encoder; one-shot decoder for small molecular graphs |

## Features

- **No generation order** — sidesteps node-ordering and error accumulation.
- **Matching-based loss** — handles graph isomorphism in supervision.

## Evolution

- **Predecessor**: autoregressive graph generation (GraphRNN).
- **Successor**: MoFlow/GraphDF flow/diffusion decoders — also one-shot, better likelihoods.
