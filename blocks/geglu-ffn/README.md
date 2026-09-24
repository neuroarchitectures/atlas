# GeGLU Feed-Forward Network

## Design Philosophy

Gated linear units improve FFN quality; the gate's activation is a free choice. GeGLU uses GELU as the gate — chosen over SwiGLU in ablations for encoder-style models where GELU is already the house activation.

## Functionality

- `FFN(x) = W_out ( GELU(W_gate x) ⊙ W_up x )`, typically with 2/3-width expansion to match parameter count of un-gated 4× FFN.

## Used By

| Model | Role |
|-------|------|
| Gemma-4 12B | Hidden 3840, intermediate 15360, 48 layers |
| ModernBERT | Hidden 768, FFN 1152, 22 encoder layers — GeGLU chosen over SwiGLU by ablation |

## Features

- **Smooth gate** — GELU's smoothness can edge out SiLU in encoder training stability.
- **Same FLOP family as SwiGLU** — drop-in activation swap.

## Evolution

- **Predecessor**: GLU (1997), T5's gated GELU FFN.
- **Sibling**: swiglu-ffn (LLaMA/Mistral default). The choice is largely empirical per model family.
