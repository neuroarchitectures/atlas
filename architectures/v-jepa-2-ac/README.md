# V-JEPA 2-AC

## Overview

V-JEPA 2-AC is the **action-conditioned world model** variant of V-JEPA 2: a lightweight predictor on top of a **frozen** V-JEPA 2 encoder is trained to roll video latents forward conditioned on robot actions and proprioception — using **under 62 hours of unlabeled robot video** — and is then used as a planner. Rolling out candidate action sequences in latent space and scoring them enables **zero-shot pick-and-place on Franka arms in two different labs**, with no robot fine-tuning.

- **Year:** 2025
- **Authors:** Assran et al. (Meta FAIR)
- **Source:** arXiv:2506.09985 — *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (V-JEPA 2-AC is the action-conditioned model introduced in the same paper)
- **Category:** DL/World-Model

## Key Characteristics

- **Frozen representation, trained predictor**: the V-JEPA 2 encoder is never fine-tuned on robot data; only the action-conditioned predictor learns — which is why <62 hours of robot video suffices.
- **Latent-space rollout**: given actions + proprioception, predict *future latents* directly — no pixel decoding, no video generation in the planning loop.
- **Planning by latent MPC**: sample action sequences, roll latents forward, score outcomes, replan.
- **Zero-shot cross-lab transfer**: policies evaluated zero-shot on Franka arms in labs whose setups differed from training data.
- Demonstrates that a video foundation model pretrained with **no actions at all** can become a robot world model with a small action-conditioned head.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Assran_et_al._2025_2506.09985.md`](references/papers/Assran_et_al._2025_2506.09985.md) (same paper as V-JEPA 2)

## Related Architectures

See `architecture.md` → Evolution section (I-JEPA → V-JEPA → V-JEPA 2 → V-JEPA 2-AC; contrast: Dreamer-style reconstructive world models).
