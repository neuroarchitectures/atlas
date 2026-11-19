# LeJEPA Identifiability

## Overview

LeJEPA Identifiability answers a deeper question than LeJEPA: **can the learned representations truly recover the world's latent variables?** The main theorem proves that LeJEPA (alignment + Gaussian regularization) **linearly recovers** the world's latent variables from nonlinear observations — up to an orthogonal transformation — for a class of worlds where latent variables evolve under stationary additive noise transitions. Crucially, the Gaussian distribution is shown to be the **unique** distribution for which this guarantee holds. The paper also proves approximate identifiability (graceful degradation when conditions are partially met) and shows that linear orthogonal identifiability enables optimal latent-space planning.

- **Year:** 2026
- **Authors:** David Klindt, Yann LeCun, Randall Balestriero (Meta FAIR / NYU)
- **Source:** arXiv:2605.26379, NeurIPS 2026 — *When Does LeJEPA Learn a World Model?*
- **Category:** DL/Theory

## Key Characteristics

- **Linear identifiability theorem**: LeJEPA recovers true world latent variables up to a linear (orthogonal) transform, from nonlinear observations.
- **Gaussian uniqueness**: among all distributions, only the Gaussian makes the identifiability guarantee hold — providing theoretical justification for SIGReg.
- **Approximate identifiability**: when conditions are not fully met, the guarantee degrades gracefully rather than catastrophically.
- **Planning implication**: linear orthogonal identifiability makes optimal latent-space planning possible.
- **Validated from 2D examples to 4096-dim latent spaces**, including pixel-based robot control and ImageNet-X natural images.
- **Key limitation**: the guarantee holds only for **stationary additive noise transitions** — real robot environments with contacts/collisions are non-stationary.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Klindt_LeCun_Balestriero_2026_2605.26379.md`](references/papers/Klindt_LeCun_Balestriero_2026_2605.26379.md)

## Related Architectures

See `architecture.md` → Evolution section (LeJEPA → LeJEPA Identifiability; sibling: PRH-JEPA provides cross-modal alignment theory).
