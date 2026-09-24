# SwiGLU Feed-Forward Network

## Design Philosophy

A gated FFN: `Swish(xW_G) ⊙ (xW_1) W_2`. The philosophy: the ReLU-FFN (`max(0, xW_1)W_2`) has no input-dependent gating; SwiGLU adds a Swish-gated branch that multiplicatively controls which features pass, improving quality at no extra depth. It is the FFN of choice for every modern decoder LLM.

## Functionality

- Three weight matrices: `W_G` (gate), `W_1` (value), `W_2` (output projection).
- `gate = Swish(x W_G)` — Swish is `x · sigmoid(x)`, smooth and non-monotonic.
- `value = x W_1`
- `output = (gate ⊙ value) W_2` — element-wise gate × value, then project down.
- Inner dimension is typically `8/3 · d_model` (chosen so total params match a 4× ReLU FFN).

## Used By

| Model | Role |
|-------|------|
| LLaMA-2/3 | FFN in every decoder block |
| Mistral / Mixtral | FFN (and expert FFN in Mixtral) |
| Qwen, Yi, Phi, Gemma | Variants |
| DeepSeek-V2/V3 | Shared expert uses SwiGLU |

## Features

- **Gating**: Input-dependent multiplicative control — better than fixed ReLU.
- **Smooth**: Swish is differentiable everywhere (unlike ReLU's kink).
- **Parameter-matched**: Inner dim tuned so param count ≈ ReLU-FFN.

## Evolution

- **Predecessor**: ReLU-FFN (original Transformer); GeLU-FFN (BERT, ViT).
- **Successor**: MoE layer (replaces the single SwiGLU with a routed mixture of SwiGLU experts).
