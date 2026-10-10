# Architecture: Beta-VAE

## Motivation

Standard VAEs learn entangled latent representations where individual dimensions do not correspond to interpretable factors of variation. A higher weight on the KL divergence term encourages a more factorized posterior.

## Core Idea

Introduce a hyperparameter beta > 1 that up-weights the KL divergence term in the VAE objective, trading off reconstruction quality for disentanglement.

## Architecture

### Overview

![beta-vae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [3, 64, 64] |
| 2 | Encoder Conv | `conv` | outChannels: 32, kernelSize: 4, stride: 2 |
| 3 | ReLU | `relu` |  |
| 4 | Encoder Conv 2 | `conv` | outChannels: 64, kernelSize: 4, stride: 2 |
| 5 | Flatten | `flatten` |  |
| 6 | Mean | `linear` | outFeatures: 32 |
| 6 | LogVar | `linear` | outFeatures: 32 |
| 7 | Reparameterize | `identity` | beta-weighted KL |
| 8 | Decoder FC | `linear` | outFeatures: 16384, inFeatures: 32 |
| 9 | Reconstruction | `output` |  |

</details>

### Components

1. **Encoder** — Convolutional encoder producing mean and log-variance of the approximate posterior. 2. **Reparameterization trick** — z = mu + sigma * epsilon, where epsilon ~ N(0,1). 3. **KL divergence with beta** — The KL term is multiplied by beta > 1, encouraging the posterior to be closer to the prior N(0,1), promoting disentanglement. 4. **Decoder** — Transposed convolution decoder reconstructing the input.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Introduce a hyperparameter beta > 1 that up-weights the KL divergence term in the VAE objective, trading off reconstruction quality for disentanglement.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: VAE. Successor: FactorVAE, TC-VAE, beta-TC-VAE.

## References

- Higgins et al. 2017
