# Shared-Expert MoE (DeepSeekMoE)

## Design Philosophy

Fine-grained expert segmentation (many small experts) increases combinatorial specialization, but routed experts alone duplicate common knowledge. Add a small number of *always-on shared experts* that every token passes through, isolating common patterns so routed experts only model the residual, routing-relevant variation.

## Functionality

- `y = Σ_{i∈topk} g_i E_i(x) + Σ_j S_j(x)` — routed experts (fine-grained, small) + shared experts (dense, always active).
- First layer of the stack is dense; per-layer routed/shared counts vary (DeepSeek-V2: 160 routed top-6 + 2 shared; GLM: 128 top-8 + 1 shared).

## Used By

| Model | Role |
|-------|------|
| DeepSeek-V2 | 160 routed (top-6) + 2 shared, expert dim 1536, hidden 5120 |
| DeepSeek-V2-Lite | Scaled-down same topology |
| DeepSeek-V3 | 256 experts top-8 + 1 shared, expert dim 2048, hidden 7168 |
| GLM-4.5-Air | 128 experts (dim 1408), top-8 + 1 shared, layer 0 dense |

## Features

- **Knowledge isolation** — routed experts stop wasting capacity on common knowledge.
- **Stability** — the shared path is always available (no routing failure mode).

## Evolution

- **Predecessor**: mixture-of-experts (Mixtral-style coarse experts); DeepSeekMoE paper (2024).
- **Companion**: auxiliary-loss-free-balancing (DeepSeek-V3) replaces balance losses for the routed part.
