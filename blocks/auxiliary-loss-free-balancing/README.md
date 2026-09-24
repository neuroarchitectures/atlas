# Auxiliary-Loss-Free Load Balancing

## Design Philosophy

Auxiliary balancing losses fight the model's specialization: they penalize exactly the expert imbalance that makes MoE useful. DeepSeek-V3 instead biases *routing scores* per expert — lower the bias of overloaded experts, raise it for underloaded ones — steering balance without touching the gradients of the model itself.

## Functionality

- Selection uses `s_i + b_i` (bias b_i, updated by a sign-based rule after each step: −γ if expert overloaded, +γ if underloaded); b_i affects only gating, not gate value or gradients.
- Complemented by sequence-wise auxiliary loss (tiny) for extreme cases.

## Used By

| Model | Role |
|-------|------|
| DeepSeek-V3 | 256 routed experts, top-8 selection, bias updated per step |

## Features

- **No gradient interference** — the main loss stays clean; balance is a control-loop on biases.
- **Specialization preserved** — experts can be genuinely unbalanced in usage if that's what data wants.

## Evolution

- **Predecessor**: switch/load balancing auxiliary losses (GShard, Switch, Mixtral aux losses).
- **Related**: Expert Choice routing (tokens routed by experts) — the other non-aux-loss balancing family.
