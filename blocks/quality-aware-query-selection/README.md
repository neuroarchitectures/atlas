# Uncertainty-Minimal Query Selection

## Design Philosophy

DETR decoder queries are learned embeddings detached from the image; RT-DETR initializes them from encoder features instead, but *which* features? Select queries by joint classification quality and localization certainty — score each candidate by classification score penalized by localization uncertainty (IoU-aware), so decoders start from the most promising, most confident anchors.

## Functionality

- Per encoder feature: cls score + uncertainty estimate; selection score minimizes joint uncertainty.
- Top-N features (from S3/S4/S5) become decoder queries with their content; positional embeddings from boxes.

## Used By

| Model | Role |
|-------|------|
| RT-DETR | Query init for the 6-layer decoder from encoder features |
| RT-DETRv3 | Retained base; adds hierarchical dense supervision on top |

## Features

- **Real-time DETR** — the query selection design is what removes DETR's训练 overhead.
- **Uncertainty-aware** — localization confidence, not just classification confidence, drives selection.

## Evolution

- **Predecessor**: two-stage DE TR (Deformable-DETR's proposal selection by top-k objectness).
- **Related**: language-guided-query-selection (Grounding DINO) — text relevance as the selection signal.
