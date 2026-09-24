# Graph Softmax

## Design Philosophy

Generating a graph token-by-token needs a *connectivity distribution* — a probability of reaching node v from node c that sums to 1 over the reachable set. Graph Softmax factors it over BFS-tree paths: `P_G(v|v_c) = Π p(node|parent)` — structure-aware, normalized, and sampleable by O(d) random walks on the tree.

## Functionality

- BFS tree rooted at v_c; probability of v = product of conditional probabilities along its path.
- Generator samples walks on the tree; discriminator scores edges `σ(α·(d_v·d_vc)+b)`; REINFORCE updates the generator.

## Used By

| Model | Role |
|-------|------|
| GraphGAN | Adversarial connectivity learning over the generator's sampled walks |

## Features

- **Proper normalization over the reachable set** — unlike naive softmax over all nodes.
- **Efficient sampling** — walk-based instead of full-distribution enumeration.

## Evolution

- **Predecessor**: flat softmax over nodes in DeepWalk-style generators.
- **Related**: one-shot-graph-decoder — the non-sequential alternative to walk-based generation.
