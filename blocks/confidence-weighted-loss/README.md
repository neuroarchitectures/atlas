# Confidence-Weighted Regression Loss

## Design Philosophy

Some pixels are simply unknowable — sky, textureless walls, motion blur. Train a *confidence head alongside the regression head* and weight the loss per pixel by predicted confidence: the network learns both the answer and where its answer can be trusted, and ambiguous regions stop dominating gradients.

## Functionality

- Loss: `Σ w_i · L_reg(x_i)` with w_i = confidence head output (DUSt3R/MASt3R: per-pixel confidence on pointmap L1-style loss).

## Used By

| Model | Role |
|-------|------|
| DUSt3R | Per-pixel confidence on pointmap regression |
| MASt3R | Confidence-weighted pointmap + matching losses |

## Features

- **Self-calibrating supervision** — the model discounts its own blind spots.
- **Confidence as output** — downstream consumers can filter by reliability.

## Evolution

- **Predecessor**: aleatoric uncertainty weighting (Kendall & Gal 2017).
- **Related**: per-region-optimal-supervision — explicit reliability classification instead of learned weighting.
