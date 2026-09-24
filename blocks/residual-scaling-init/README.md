# Residual-Depth-Scaled Init / muP Scaled Residuals

## Design Philosophy

Deep residual stacks accumulate variance; without compensation, activations explode with depth. Two closely-related stabilizers: scale residual-path weights by 1/√N at initialization (GPT-2), or scale the *residual branches* by depth-dependent constants throughout training (MiniCPM's muP-style: embedding output × scale_emb, each residual branch × scale_depth/√num_layers) — Magneto adds a LayerNorm after each sublayer with theoretically derived init scaling for the same goal.

## Functionality

- GPT-2: init c_proj weights with std 0.02/√N (N = residual layer count).
- MiniCPM: `x ← x + (scale_depth/√num_layers) · branch(x)`, scale_emb=12, scale_depth/√40; tied embeddings.
- Magneto: post-sublayer LN + variance-preserving init for stable multimodal pretraining.

## Used By

| Model | Role |
|-------|------|
| GPT-2 | 1/√N c_proj init across 12–48 residual layers |
| MiniCPM-2B | muP-style scaled residuals, 40 layers, hidden 2304 |
| LWM | Magneto sublayer scaling in the 24-layer multimodal decoder |

## Features

- **Depth-stable by construction** — variance preserved regardless of stack height.
- **Zero inference cost** — scaling folds into weights or is absorbed by the next norm.

## Evolution

- **Predecessor**: GPT-2 init (2020); DeepNet's 2N-scaled residual.
- **Related**: nGPT's slerp-residual-update — renormalization-based stability instead of init-based.
