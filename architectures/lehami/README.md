# LeHami

## Overview

LeHami discovers a critical failure mode of unconstrained JEPA predictors: **predictor contraction** — in autoregressive rollout, each step's prediction slightly contracts the latent distribution, and these contractions accumulate over time, causing the distribution to collapse. The solution: replace the unconstrained predictor with a **Hamiltonian flow** acting on a structured latent state (q, p), stepped by an **exact symplectic integrator**. This guarantees: (1) **exact volume preservation** in phase space at every step (excluding uniform contraction, asymptotic fixed points, and state merging — the three collapse modes), (2) **continuous-time stepping** (a single trained model integrates at any step size), and (3) **time reversibility** (flipping time enables backward rollout for safety verification).

- **Year:** 2026
- **Authors:** Gabriel Yalovetzky, Randall Balestriero, Yann LeCun (Meta FAIR / NYU)
- **Source:** arXiv:2610.15267 — *LeHami: Symplectic Latent Dynamics for World Models*
- **Category:** DL/World-Model

## Key Characteristics

- **Predictor contraction identified**: unconstrained JEPA predictors contract the latent distribution during autoregressive rollout, leading to gradual collapse.
- **Hamiltonian flow replacement**: the predictor is replaced by H(q,p) acting on a structured (q, p) phase space, where q = generalized positions, p = generalized momenta.
- **Exact symplectic integration**: the integrator is exactly symplectic (a symplectic diffeomorphism) at any step size and any parameters — volume is **exactly** preserved, not approximately.
- **Three collapse modes excluded**: Proposition 1 proves the symplectic step excludes (1) uniform contraction, (2) convergence to asymptotically stable fixed points, (3) merging of distinct states.
- **Continuous-time**: a single trained model can be integrated at any step size — no retraining for different temporal resolutions.
- **Time-reversible**: flipping the time sign enables backward rollout, useful for safety guarantees and counterfactual reasoning.
- **Validated on pendulum environments** — generalization to multi-body interactions is future work.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Yalovetzky_et_al._2026_2610.15267.md`](references/papers/Yalovetzky_et_al._2026_2610.15267.md)

## Related Architectures

See `architecture.md` → Evolution section (LeWM → LeHami; sibling: LpWM provides sparse dynamics; contrast: V-JEPA 2-AC uses unconstrained predictor).
