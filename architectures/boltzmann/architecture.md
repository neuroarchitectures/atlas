# Architecture: Boltzmann Machine

## Motivation

The Hopfield network is deterministic and limited in capacity. The Boltzmann Machine introduces stochastic units and hidden units, enabling the network to learn internal representations and model probability distributions over the visible units.

## Core Idea

A fully connected network of binary stochastic units (visible + hidden) with symmetric weights. The network defines a Boltzmann distribution P(v) ∝ exp(-E(v)/T). Training uses contrastive divergence: clamp visible units to data, let hidden units settle, then unclamp and let the network freely sample. Weight updates reduce the difference between data and model statistics.

## Architecture

### Overview

![boltzmann architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Visible Units | `input` | shape: [1, 100] |
| 2 | Visible-Hidden Weights | `custom` | type: symmetric_weights, sizeV: 100, sizeH: 50 |
| 3 | Hidden Units | `custom` | type: stochastic_binary, size: 50 |
| 4 | Energy Function | `custom` | type: boltzmann_energy |
| 5 | Gibbs Sampling | `custom` | type: gibbs_sampling, steps: 1000 |
| 6 | Contrastive Divergence | `custom` | type: cd_update |
| 7 | Learned Distribution | `output` |  |

</details>

A fully connected network of binary stochastic units (visible + hidden) with symmetric weights. The network defines a Boltzmann distribution P(v) ∝ exp(-E(v)/T). Training uses contrastive divergence: clamp visible units to data, let hidden units settle, then unclamp and let the network freely sample. Weight updates reduce the difference between data and model statistics.

### Components

2. **Visible-Hidden Weights** (`custom`, scope: `model`) — Params: type: symmetric_weights, sizeV: 100, sizeH: 50
3. **Hidden Units** (`custom`, scope: `model`) — Params: type: stochastic_binary, size: 50
4. **Energy Function** (`custom`, scope: `model`) — Params: type: boltzmann_energy
5. **Gibbs Sampling** (`custom`, scope: `training`) — Params: type: gibbs_sampling, steps: 1000
6. **Contrastive Divergence** (`custom`, scope: `training`) — Params: type: cd_update

### Data Flow

The architecture processes input through a sequence of 7 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Boltzmann Machine.

## Evolution

The Boltzmann Machine extends Hopfield Networks with stochastic units and hidden layers. Successors: Restricted Boltzmann Machine (RBM, no intra-layer connections), Deep Belief Networks (stacked RBMs), and energy-based models in modern deep learning.

## Source

- **Paper:** Cognitive Science 1985
- **Year:** 1985
- **Authors:** Ackley et al.
