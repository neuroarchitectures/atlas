# Memory Token Segment Recurrence

## Design Philosophy

Transformers have fixed context; recurrent memory restores unbounded input without changing the architecture. Append a few learnable *memory tokens* to each segment's input; the self-attention layers read and update them, and the output memory tokens become the input memory of the next segment — gradients flow through them (truncated BPTT).

## Functionality

- Segment: `[mem_in, tokens] → transformer → [mem_out, outputs]`; `mem_in ← mem_out` for the next segment.
- Constant memory/compute per segment; scales to 50M-token streams (RMT, ARMT: memory stored as key-value associations).

## Used By

| Model | Role |
|-------|------|
| RMT | 10 memory tokens, 150-token segments |
| ARMT | Associative memory variant; BABILong 79.9% at 50M tokens |

## Features

- **Zero architecture change** — the vanilla transformer body is untouched.
- **Linear scaling in sequence length** with global information flow.

## Evolution

- **Predecessor**: Transformer-XL segment recurrence (hidden-state caching).
- **Successor**: compressive/adaptive memory variants; in-context state competes with SSMs (selective-ssm).
