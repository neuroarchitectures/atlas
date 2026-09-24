# Region-Text Contrastive Head

## Design Philosophy

A fixed classifier head cannot recognize unseen categories. Score regions directly by dot-product similarity between region features and text sub-sentence features — classes become *vectors in language space*, so new categories need zero new parameters.

## Functionality

- Region features (from decoder/encoder) vs sub-sentence text features: logits = `τ · r_i · t_j`.
- Trained with focal-style contrastive loss + box L1/GIoU; contrastive alignment also used in pre-training alignment.

## Used By

| Model | Role |
|-------|------|
| Grounding DINO | Class head (with box head); bidirectional enhancer feeds region/text features |

## Features

- **Open-vocabulary classification** — head size independent of category count.
- **Token-level grounding** — sub-sentence features enable phrase-level boxes.

## Evolution

- **Predecessor**: CLIP's image-text contrastive-dual-encoder (image-level, not region-level).
- **Successor**: OWLv2, Grounding DINO 1.5/1.6; SAM 3 presence-head handles the "absent concept" case.
