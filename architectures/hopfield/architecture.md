# Architecture: Hopfield Network

## Motivation

How can a neural network store and retrieve memories? Hopfield introduced a recurrent network with symmetric connections whose dynamics minimize an energy function, storing patterns as local minima (attractors) of the energy landscape.

## Core Idea

A fully connected recurrent network with symmetric weights (w_ij = w_ji) and binary units. Patterns are stored via Hebbian learning (w_ij = Σ_patterns x_i^p x_j^p). Retrieval works by setting the network state to a noisy version of a stored pattern and letting it converge to the nearest attractor via asynchronous updates.

## Architecture

### Overview

![hopfield architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input Pattern | `input` | shape: [1, 100] |
| 2 | Weight Matrix W | `custom` | type: symmetric_weights, size: 100 |
| 3 | Net Input | `linear` | outFeatures: 100, inFeatures: 100 |
| 4 | Sign Activation | `custom` | type: sign_activation |
| 5 | Asynchronous Update | `custom` | type: async_update |
| 6 | Energy Check | `custom` | type: energy_function |
| 7 | Converged? | `custom` | type: convergence_check |
| 8 | Retrieved Pattern | `output` |  |

</details>

A fully connected recurrent network with symmetric weights (w_ij = w_ji) and binary units. Patterns are stored via Hebbian learning (w_ij = Σ_patterns x_i^p x_j^p). Retrieval works by setting the network state to a noisy version of a stored pattern and letting it converge to the nearest attractor via asynchronous updates.

### Components

2. **Weight Matrix W** (`custom`, scope: `memory`) — Params: type: symmetric_weights, size: 100
3. **Net Input** (`linear`, scope: `update`) — Params: outFeatures: 100, inFeatures: 100
4. **Sign Activation** (`custom`, scope: `update`) — Params: type: sign_activation
5. **Asynchronous Update** (`custom`, scope: `dynamics`) — Params: type: async_update
6. **Energy Check** (`custom`, scope: `dynamics`) — Params: type: energy_function
7. **Converged?** (`custom`, scope: `dynamics`) — Params: type: convergence_check

### Data Flow

The architecture processes input through a sequence of 8 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Hopfield Network.

## Evolution

The Hopfield Network introduced attractor-based computation and energy landscapes. Successors: Boltzmann Machine (stochastic version), RBM, Modern Hopfield Networks (dense associative memories with exponential capacity), and continuous Hopfield networks used in attention.

## Source

- **Paper:** PNAS 1982
- **Year:** 1982
- **Authors:** Hopfield
