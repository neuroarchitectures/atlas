# Entire-Space Multi-Task Towers

## Design Philosophy

CVR models trained only on clicks suffer sample-selection bias: clicks are a biased subsample of impressions. ESMM instead trains pCTR and pCVR towers *jointly* with the identity pCTCVR = pCTR × pCVR supervised over **all impressions** — the CVR tower receives gradient through the product, covering the entire space without click-only labels.

## Functionality

- Shared embedding table → two MLP towers (pCTR, pCVR).
- Losses: `L(pCTR, y_click) + L(pCTR × pCVR, y_conversion)` over every impression.

## Used By

| Model | Role |
|-------|------|
| ESMM | Entire-space CVR estimation in industrial ranking |

## Features

- **Selection-bias eliminated** — CVR learns from impression-space supervision.
- **Product objective = free causal structure** — CVR cannot exceed CTR by construction.

## Evolution

- **Predecessor**: click-conditioned CVR training; multi-gate-moe towers.
- **Successor**: ESM² (elongated tower chains); delayed-feedback variants.
