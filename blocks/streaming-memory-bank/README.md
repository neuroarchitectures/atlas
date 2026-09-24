# Streaming Memory Bank

## Design Philosophy

Video segmentation needs all prior evidence, but keeping every frame's features is infeasible. SAM 2 maintains a *fixed-size* memory: compact mask-conditioned frame tokens (spatial) plus object pointers (a single vector), written by a memory encoder and attended via cross-attention — enabling streaming inference with bounded compute per frame.

## Functionality

- Memory encoder: conv over predicted mask + frame features → few tokens per frame; FIFO over recent frames + all prompted frames.
- Object pointers: single attention-weighted vector per object.
- Memory attention (with RoPE) conditions the current frame's features on the bank.

## Used By

| Model | Role |
|-------|------|
| SAM 2 | Streaming video segmentation; masklets propagate via memory attention |

## Features

- **Bounded per-frame cost** — the key to real-time streaming video segmentation.
- **Mask-conditioned memory** — memory stores *what was segmented*, not raw features.

## Evolution

- **Predecessor**: XMem/STM memory banks (video object segmentation).
- **Related**: query-memory-queue (StreamPETR) — query vectors instead of frame tokens as temporal state.
