# Joint Detection-Tracking Decoder Queries

## Design Philosophy

Detection-then-association pipelines propagate IDs by a separate matching step. SAM 3's decoder instead makes object queries *double as track identities*: the same query set produces per-frame detections and persists across frames, so association is implicit in query continuity.

## Functionality

- DETR-style object queries per concept; queries producing a detection in frame t attend to (carry) their state into frame t+1, yielding consistent identity.
- Combined with a presence head that gates whether the concept exists at all.

## Used By

| Model | Role |
|-------|------|
| SAM 3 | Unified detector-tracker for promptable concept segmentation in image and video |

## Features

- **No separate association module** — identity is architectural, not post-hoc matching.
- **Unifies image and video** in one decoder.

## Evolution

- **Predecessor**: Tracktor, MOTR (track queries) — the direct ancestor in MOT literature.
- **Related**: query-memory-queue (StreamPETR) — temporal query state for 3D detection; streaming-memory-bank (SAM 2).
