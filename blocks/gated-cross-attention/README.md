# Gated Cross-Attention (Multimodal Fusion)

## Design Philosophy

Splice cross-attention layers into a frozen LLM, with **tanh gates** initialized to zero, so the model starts as exactly the frozen LLM and learns to use the new modality gradually. The philosophy: training a frozen LLM + new cross-attention is unstable if the new layers immediately perturb the LLM's representations; zero-init gates make the start a pure identity, and the model opens the gates as it learns.

## Functionality

- Insert cross-attention layers between the frozen LLM's blocks.
- Each inserted layer: `x → cross-attention(Q=x, K=V=visual_tokens) → tanh(gate) → add(x)`.
- **Gate**: `gate = tanh(α) · α`, where `α` is a learnable scalar init to 0.
- At init, `gate = 0`, so the layer is identity → the model is exactly the frozen LLM.
- As training proceeds, `α` grows and the cross-attention contributes.

## Used By

| Model | Role |
|-------|------|
| Flamingo | Gated cross-attention to Perceiver-resampled vision features |
- The "freeze the LLM, splice in a modality" pattern.

## Features

- **Stable training**: Zero-init gates prevent catastrophic perturbation of the frozen LLM.
- **Modality injection**: Cross-attention is the bridge; the gate controls how much.
- **Frozen LLM**: Only the cross-attention layers (and gates) are trained.

## Evolution

- **Predecessor**: Cross-attention (Transformer decoder); adapter layers (zero-init residual additions).
- **Successor**: Prefix-token injection (LLaVA, no cross-attention); the gated-insertion pattern influenced later MLLMs.
