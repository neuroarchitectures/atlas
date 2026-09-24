# Edge-Feature Attention Injection

## Design Philosophy

Standard attention ignores edge attributes; Graphormer adds only learned distance biases. Multiply *projected edge embeddings* directly into the pairwise attention scores, and maintain a separate node-symmetric edge pipeline (own FFN and residual) so edge semantics are updated, not just consumed.

## Functionality

- Attention logit: `q_i·k_j + ŵ_ij · (E e_ij)` with learned edge projection E.
- Parallel edge channel: edges have their own FFN/residual stream updated each block.

## Used By

| Model | Role |
|-------|------|
| Graph Transformer | Edge-aware attention + edge FFN pipeline (expand 2d) |

## Features

- **Content-sensitive edge bias** — the edge's meaning, not just its existence, modulates attention.
- **Edge representations evolve** — enabling edge prediction tasks.

## Evolution

- **Predecessor**: Graphormer spatial encoding; relational attention (R-AT).
- **Related**: graph-propagation-attention — full three-path unification of node/edge propagation.
