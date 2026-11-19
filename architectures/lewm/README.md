# LeWM (LeWorldModel)

## Overview

LeWorldModel (LeWM) is the first JEPA world model that trains **end-to-end from raw pixels** in a stable manner, with no pretrained encoder, no EMA teacher, no stop-gradient, and no auxiliary supervision. Using only two loss terms — next-embedding prediction (MSE) and SIGReg regularization — LeWM reduces the loss hyperparameter count from 6 (in prior end-to-end methods) to 1. With just 15M parameters trained on a single GPU in hours, it achieves competitive planning performance across 2D and 3D control tasks at 48× the speed of baseline world models.

- **Year:** 2026
- **Authors:** Florian Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero (Meta FAIR / Inria)
- **Source:** arXiv:2603.19312 — *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*
- **Category:** DL/World-Model

## Key Characteristics

- **End-to-end from pixels**: no pretrained encoder, no frozen features — the encoder and predictor are trained jointly from raw pixel input.
- **Only 2 loss terms**: next-embedding MSE + SIGReg — down from 6+ in prior methods.
- **No heuristics**: no stop-gradient, no EMA, no predictor asymmetry, no auxiliary losses.
- **15M parameters, single GPU, hours of training** — demonstrates that JEPA world models don't require massive compute.
- **48× faster planning** than baseline world models.
- **Latent space encodes physical structure spontaneously** — verified via physical quantity probing.
- **Surprise evaluation** confirms the model reliably detects physically impossible events.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Maes_et_al._2026_2603.19312.md`](references/papers/Maes_et_al._2026_2603.19312.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA 2-AC → LeWM; siblings: LeJEPA provides SIGReg; successors: LeHami adds symplectic dynamics, LpWM adds sparse representations).
