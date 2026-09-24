# ConvGRU Recurrent Update Operator

## Design Philosophy

Dense fields (flow, disparity) refine best by *iteration with memory*: a convolutional GRU holds per-pixel hidden state, ingesting context features + correlation lookups + the current estimate each step, and outputting a residual Δ. Inference depth = number of iterations — an accuracy/compute dial with no architecture change.

## Functionality

- `h_{t+1} = ConvGRU(h_t, [context, corr_lookup, x_t])`; `x_{t+1} = x_t + Δ(x_t, h_{t+1})`.
- Weight-tied across iterations (~2.7M params in RAFT); flow initialized to zero (or regressed — SEA-RAFT).

## Used By

| Model | Role |
|-------|------|
| RAFT | Iterative flow refinement from correlation + context |
| RAFT-Stereo | All-pairs correlation + ConvGRU disparity refinement |
| SEA-RAFT | Same loop with mixture-of-laplace-loss |
| GMFlow | Single refinement pass (loop unrolled once) |
| FlowFormer | ConvGRU refinement over cost-memory retrieval |
| IGEV | ConvGRU iterations indexing the geometry-encoding-volume |

## Features

- **Anytime inference** — 32 iterations for accuracy, 12 for speed.
- **Explicit state** — hidden state carries occlusion/rigidity cues across iterations.

## Evolution

- **Predecessor**: PWC-Net (single-shot warping + refinement).
- **Successor**: transformers over spacetime tokens (CoTracker3's joint-spacetime-attention) replace the GRU in tracking; GMFlow shows one iteration can suffice.
