# Architecture: Restricted Boltzmann Machine

## Motivation

The full Boltzmann Machine is hard to train because connections between all units make Gibbs sampling very slow. The RBM removes intra-layer connections (visible-visible and hidden-hidden), making the layers conditionally independent given each other, which enables efficient block Gibbs sampling.

## Core Idea

A bipartite graph with visible units V and hidden units H, connected only by inter-layer weights W. Given the visible units, all hidden units are conditionally independent (and vice versa), enabling fast sampling. Training uses contrastive divergence (CD-k): clamp visible to data, sample hidden, reconstruct visible, resample hidden, and update weights.

## Architecture

### Overview

![rbm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Visible Units v | `input` | shape: [1, 784] |
| 2 | W (v→h) | `linear` | outFeatures: 128, inFeatures: 784 |
| 3 | Bias b | `custom` | type: bias, size: 128 |
| 4 | vW + b | `add` |  |
| 5 | P(h|v) | `sigmoid` |  |
| 6 | Sample h | `custom` | type: bernoulli_sample |
| 7 | W^T (h→v') | `linear` | outFeatures: 784, inFeatures: 128 |
| 8 | Bias a | `custom` | type: bias, size: 784 |
| 9 | hW^T + a | `add` |  |
| 10 | P(v'|h) | `sigmoid` |  |
| 11 | CD Update | `custom` | type: contrastive_divergence |
| 12 | Reconstruction | `output` |  |

</details>

A bipartite graph with visible units V and hidden units H, connected only by inter-layer weights W. Given the visible units, all hidden units are conditionally independent (and vice versa), enabling fast sampling. Training uses contrastive divergence (CD-k): clamp visible to data, sample hidden, reconstruct visible, resample hidden, and update weights.

### Components

2. **W (v→h)** (`linear`, scope: `model`) — Params: outFeatures: 128, inFeatures: 784
3. **Bias b** (`custom`, scope: `model`) — Params: type: bias, size: 128
4. **vW + b** (`add`, scope: `model`) — Params: none
5. **P(h|v)** (`sigmoid`, scope: `model`) — Params: none
6. **Sample h** (`custom`, scope: `model`) — Params: type: bernoulli_sample
7. **W^T (h→v')** (`linear`, scope: `model`) — Params: outFeatures: 784, inFeatures: 128
8. **Bias a** (`custom`, scope: `model`) — Params: type: bias, size: 784
9. **hW^T + a** (`add`, scope: `model`) — Params: none
10. **P(v'|h)** (`sigmoid`, scope: `model`) — Params: none
11. **CD Update** (`custom`, scope: `training`) — Params: type: contrastive_divergence

### Data Flow

The architecture processes input through a sequence of 12 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Restricted Boltzmann Machine.

## Evolution

RBM simplifies the Boltzmann Machine by removing intra-layer connections. Successors: Deep Belief Networks (stacked RBMs trained greedily), Deep Boltzmann Machines, and modern energy-based models. RBMs were key to pre-training deep networks before ReLU and BatchNorm.

## Source

- **Paper:** Parallel Distributed Processing 1986
- **Year:** 1986
- **Authors:** Smolensky
