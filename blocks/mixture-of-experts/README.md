# Mixture-of-Experts (MoE) Layer

## Design Philosophy

Replace the single FFN with a set of `E` expert FFNs and a **router** that selects the top-k experts per token. The philosophy: decouple parameter count from compute — a 47B model can have only 13B active per token. Different experts specialize, and sparse routing means the model's capacity grows without proportional inference cost.

## Functionality

- **Router**: A linear `d_model → numExperts` producing logits per token.
- **Top-k selection**: Keep the top-k experts (e.g., top-2 in Mixtral, top-6 in DeepSeek-V2) per token.
- **Weighted sum**: `output = Σ_{i in top-k} softmax(logit_i) · Expert_i(x)`.
- **Load balancing**: Auxiliary loss penalizes uneven expert usage to prevent collapse.
- **Shared experts** (DeepSeek): A few always-on experts handle common patterns; routed experts handle specialization.

## Used By

| Model | Role |
|-------|------|
| Mixtral 8×7B | 8 experts, top-2, 13B active of 47B |
| DeepSeek-V2 | 160 routed experts, top-6 + shared |
| DeepSeek-V3 | 256 experts, scaled to 671B |
| GShard, Switch Transformer | Early MoE at scale |

## Features

- **Capacity vs. compute**: Total params scale freely; active compute stays bounded by top-k.
- **Specialization**: Experts learn different functions (e.g., syntax, math, code).
- **Load balancing**: Critical to avoid all tokens routing to one expert.

## Evolution

- **Predecessor**: Dense FFN; Mixture-of-Experts (Jacobs, 1991) at the model level.
- **Successor**: Fine-grained MoE (DeepSeek's many slim experts); shared + routed expert mixtures; MoE in attention (MoA).
