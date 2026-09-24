# Perceiver Resampler

## Design Philosophy

A fixed set of learned latents cross-attend variable-length vision features, producing a constant number of visual tokens. The philosophy: images (or videos) have variable token counts, but the downstream LLM wants a fixed prefix — the resampler compresses any input to a constant 64 tokens. Unlike the Q-Former, it's typically inserted into a frozen LLM via gated cross-attention.

## Functionality

- **Learned latents**: 64 trainable query vectors.
- **Cross-attention**: Latents cross-attend the frozen vision encoder's output (NFNet / CLIP).
- **Output**: 64 visual tokens, fed to the LLM.
- **tanh-gated insertion** (Flamingo): The cross-attention layers are spliced into the frozen LLM with `tanh` gates init to zero, so the model starts as exactly the frozen LLM and learns to use vision gradually.

## Used By

| Model | Role |
|-------|------|
| Flamingo | The visual resampler + gated cross-attention into frozen LLM |
| Perceiver / Perceiver IO | The original learned-latents cross-attention |

## Features

- **Constant output**: 64 tokens regardless of input size.
- **Stable training**: tanh-gated insertion starts at zero contribution.
- **Frozen backbone**: Only the resampler (and gates) are trained.

## Evolution

- **Predecessor**: Perceiver (learned latents); BLIP-2's Q-Former (similar idea).
- **Successor**: LLaVA's MLP projector (simpler); the "resampler" pattern influenced later MLLM bridges.
