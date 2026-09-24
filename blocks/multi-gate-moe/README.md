# Multi-Gate Mixture-of-Experts (MMoE)

## Design Philosophy

Give each task a **softmax gate** that mixes a shared pool of experts its own way. The philosophy: a single shared bottom forces conflicting objectives to share a representation (negative transfer); MMoE lets disagreeing tasks lean on different experts without a hard split. The default multi-task ranker behind feed and video systems.

## Functionality

- **Shared bottom**: Input features → a shared bottom MLP.
- **N expert MLPs**: Each expert is a separate MLP fed by the shared bottom.
- **Per-task gates**: For each task `t`, a softmax gate `g_t(x) = softmax(W_t · x)` produces mixture coefficients over the N experts.
- **Task-specific mixture**: `mixture_t = Σ_i g_{t,i} · Expert_i(x)`.
- **Towers**: One MLP head per objective, fed by its mixture.

## Used By

| Model | Role |
|-------|------|
| MMoE (Google) | Multi-task ranking (feed, video) |
| PLE (Progressive Layered Extraction) | Hardens soft gating into task-specific vs shared expert groups |

## Features

- **Soft expert sharing**: Tasks share experts but through different gates.
- **Avoids negative transfer**: Conflicting tasks can route to different experts.
- **Per-task gate**: Each task gets its own mixture.

## Evolution

- **Predecessor**: Shared-bottom multi-task (one MLP for all tasks); MoE (model-level, Shazeer).
- **Successor**: PLE (explicit task-specific + shared expert groups); the pattern is now standard in multi-task ranking.
