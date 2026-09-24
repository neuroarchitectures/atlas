# Contrastive Denoising Queries (CDN)

## Design Philosophy

DETR's bipartite matching makes early training unstable — the model chases a moving target. CDN injects *known-answer* queries: noised ground-truth boxes, labeled with their class. Small noise ("positive" group) teaches box refinement; large noise ("negative" group) is supervised as ∅, teaching the model to suppress duplicates — the cause of DETR's double predictions.

## Functionality

- Two query groups per GT box: positives (small perturbation) and negatives (large perturbation past the boundary).
- Positives supervised with class + box losses; negatives supervised with "no object". Training-only; stripped at inference.

## Used By

| Model | Role |
|-------|------|
| DINO | CDN query group alongside mixed-selection matching queries |

## Features

- **Static supervision anchor** — removes the matching cold-start instability.
- **Duplicate suppression by construction** — negatives near GT teach the ∅ boundary.

## Evolution

- **Predecessor**: DN-DETR (denoising groups); the instability diagnosis of bipartite matching.
- **Successor**: DINO's mixed query selection + look-forward-twice complete the recipe; RT-DETR achieves stability via query selection instead.
