# Recurrent Temporal BEV Attention

## Design Philosophy

A single-frame BEV has no velocity and cannot see behind occlusions. Let the BEV queries attend to the *previous frame's BEV feature map*, aligned by ego-motion, as a deformable self-attention across time — the BEV grid itself becomes the recurrent hidden state.

## Functionality

- Shift previous BEV features by ego-motion (translation/rotation) between timestamps.
- BEV queries perform deformable attention over the shifted map (sampled offsets around expected locations) fused with current-frame spatial attention.

## Used By

| Model | Role |
|-------|------|
| BEVFormer | Temporal self-attention over ego-motion-aligned previous BEV; recurrent state across frames |

## Features

- **Velocity from recurrence** — moving objects leave motion cues in the aligned difference.
- **Occlusion reasoning** — previously-observed geometry persists in the state.

## Evolution

- **Predecessor**: single-frame BEV lifting (LSS/DETR3D).
- **Successor**: query-memory-queue (StreamPETR) — object queries replace the dense grid as the recurrent state, cutting memory.
