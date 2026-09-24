# Token Shift

## Design Philosophy

Gating mechanisms need past-timestep information, but the cheapest source isn't a recurrent state — it's a *one-step shift of the input itself*. Interpolate the current input with the previous timestep's input, applied independently before every projection: temporal context at zero parameters.

## Functionality

- `x'_t = α · x_t + (1-α) · x_{t-1}` (RWKV: 50/50 mix), applied per channel before R/K/V projections in time-mixing and R/K in channel-mixing.

## Used By

| Model | Role |
|-------|------|
| RWKV | Pre-shift for time-mixing (R/K/V) and channel-mixing (R/K) in every block |

## Features

- **Free temporal context** — no state, no recurrence, fully parallel.
- **Per-projection independence** — each projection sees its own shifted view.

## Evolution

- **Predecessor**: temporal convolution kernels (TrellisNet), time-delay embeddings.
- **Related**: selective-ssm's state carries longer history; token-shift covers the 1-step case in linear-cost blocks.
