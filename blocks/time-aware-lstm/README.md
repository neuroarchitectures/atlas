# Time-Aware LSTM

## Design Philosophy

User behavior arrives at *irregular* intervals, but a standard LSTM sees only the order of events. Feed the elapsed time gap between actions into the input and forget gates, so memory decays according to real elapsed time, not step count.

## Functionality

- Gates additionally consume Δt (log-scaled time gaps): `i_t = σ(W_i[h_{t-1}, x_t, Δt])`, similarly forget `f_t`.
- Short-term interest branch follows the time-decayed memory; fusion with a long-term branch via a sigmoid gate.

## Used By

| Model | Role |
|-------|------|
| SLi-Rec | 32→64 single-layer time-aware LSTM; short-term branch of the interest model |

## Features

- **Irregular sampling native** — no resampling or bucketing of event times.
- **Gate-level time injection** — time modulates *memory*, not just features.

## Evolution

- **Predecessor**: Time-LSTM (2017), Phased LSTM (sensor irregular sampling).
- **Related**: hawkes-process-embedding — point-process modeling of the same irregular timing.
