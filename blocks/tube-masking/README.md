# Tube Masking

## Design Philosophy

Random 2D masking lets a video model cheat: adjacent frames reveal the masked patch. Mask entire *spatio-temporal tubes* — the same spatial region across time — so temporal reasoning is mandatory and cross-frame leakage is impossible.

## Functionality

- Choose spatial regions per clip; mask the full temporal extent of each region (tubelet-aligned).
- VideoMAE pushes the ratio extremely high (~90%) since tubes make reconstruction harder.

## Used By

| Model | Role |
|-------|------|
| V-JEPA / V-JEPA-2 | Tube masking for latent prediction objectives |
| VideoMAE | Very-high-ratio tube masking for pixel reconstruction |

## Features

- **No temporal shortcut** — the defining fix over 2D MAE naively applied to video.
- **High tolerable mask ratio** → large efficiency gains.

## Evolution

- **Predecessor**: MAE random 2D masking; VideoMAE's tube redesign (2022).
- **Related**: cascade-masking (VideoPrism) — ratio scheduling across the corpus.
