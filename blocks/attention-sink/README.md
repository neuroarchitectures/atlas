# Learned Attention Sink

## Design Philosophy

Softmax attention must sum to 1, so excess probability mass piles onto the first tokens ("sinks") — and in sliding-window models this distorts the window. Give each head *learnable sink logits* — virtual keys outside the sequence that absorb surplus mass — restoring clean within-window attention.

## Functionality

- Per head: `s_j = q·k_j`, plus a learned sink logit s_0; softmax over [sink, window keys].
- Streaming decode keeps sinks resident so the window never "rolls over" a sink token.

## Used By

| Model | Role |
|-------|------|
| gpt-oss-120b / gpt-oss-20b | Learned per-head sink logits in sliding-window layers (128-window, alternating with full attention) |

## Features

- **Diagnosis turned mechanism** — the StreamingLLM sink phenomenon, made trainable.
- **Stable infinite-length streaming** — window can slide without quality collapse.

## Evolution

- **Predecessor**: StreamingLLM (2023) observed sink tokens; attention-with-zero scores workarounds.
- **Related**: sliding-window-attention — sinks are the stability patch for long streaming windows.
