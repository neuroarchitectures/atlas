# Global Optimal Localization Self-Distillation (GoLSD)

## Design Philosophy

D-FINE's decoder layers refine box distributions progressively — so the *last* layer's refined distributions are the best available teacher for earlier layers. Distill them backwards, weighted by each layer's localization quality, giving early layers a global view of the final localization instead of only their local gradient.

## Functionality

- Final decoder layer's distributions as targets; earlier layers distill toward them with per-layer quality weighting.
- Training-only branch across the 6 decoder layers; stripped at inference.

## Used By

| Model | Role |
|-------|------|
| D-FINE | Self-distillation across decoder layers alongside fine-grained-distribution-refinement |

## Features

- **Self-teacher, no extra model** — the final layer is free supervision.
- **Quality-weighted** — distillation strength follows localization confidence.

## Evolution

- **Predecessor**: self-distillation (born-again networks); D-FINE's FDR refinement.
- **Related**: look-forward-twice (DINO) — the box-coordinate version of the same "teach early layers" idea.
