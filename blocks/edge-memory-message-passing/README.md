# Edge Memory Message Passing

## Design Philosophy

Node-centric message passing discards the *history of interactions* between a pair of nodes. Keep edge embeddings as latent memory across message-passing steps, reuse them as skip connections, and integrate with implicit-Euler stepping for stable dynamics — the edges remember.

## Functionality

- Edge embeddings e_ij persist across sub-time-steps; each step's messages read and update the memory.
- Implicit Euler integration over the interaction dynamics; memory reused as per-step skip connections.

## Used By

| Model | Role |
|-------|------|
| PI-GNN | Physics-informed interaction network: edge memory + implicit Euler sub-time-stepping |

## Features

- **Pairwise history retention** — interaction dynamics, not just current state.
- **Stable long rollouts** — implicit integration damps the stiffness of physical systems.

## Evolution

- **Predecessor**: relation networks,Interaction Networks (Battaglia 2016) — per-step edge computation without memory.
- **Related**: temporal-graph-networks (edge memories in discrete-time TGNNs).
