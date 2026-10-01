# Top-K Routing (MoE)

## Design Philosophy

In Mixture-of-Experts, the router selects the top-k experts for each token. Only the selected experts are evaluated, enabling conditional computation with sparse activation.

## Functionality

Router: r(x) = W_r * x, produces logits for E experts. Top-k selection: keep the k highest scores, set rest to -inf. Softmax over selected experts. Token is sent to selected experts only.

## Used By

Switch Transformer | GShard | Mixtral | DeepSeek-MoE

## Features

- **Sparse activation**: Only k experts per token, reducing FLOPs.
- **Load balancing**: Auxiliary loss prevents expert collapse.
- **Capacity factor**: Limits tokens per expert for load balancing.

## Evolution

Predecessor: Soft MoE (all experts). Successor: Expert choice routing, soft MoE.
